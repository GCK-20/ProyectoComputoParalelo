"""
Leemos los mapas mensuales (GeoTIFF de Earth Engine) y construye, en paralelo,
la tabla para predecir la temperatura de cada pixel en el mes t a partir de
los mapas de calor de meses anteriores.

Paralelismo de datos: la imagen se parte en bloques de filas y cada proceso
arma las variables de su bloque (multiprocessing, sin joblib).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import multiprocessing as mp
import numpy as np
import pandas as pd

ANIO_INICIO = 2015
SIN_DATO = -9999
RETRASOS = [1, 2, 3, 12]          # meses anteriores que se usan como entrada

# Variables de entrada (X) y variable a predecir (y)
COLUMNAS_X = ["LST_t1", "LST_t2", "LST_t3", "LST_t12", "NDVI_t1",
              "ALTITUD", "lat", "lon", "mes_sin", "mes_cos"]
COLUMNA_Y = "LST"


def leer_mapas(ruta_lst, ruta_ndvi, ruta_alt):
    """Regresa LST y NDVI como arreglos (meses, filas, columnas), la altitud
    (filas, columnas) y las coordenadas de cada pixel."""
    import rasterio
    with rasterio.open(ruta_lst) as f:
        lst = f.read().astype("float32")
        transform = f.transform
    with rasterio.open(ruta_ndvi) as f:
        ndvi = f.read().astype("float32")
    with rasterio.open(ruta_alt) as f:
        alt = f.read(1).astype("float32")
    for a in (lst, ndvi, alt):
        a[a == SIN_DATO] = np.nan
    filas, cols = alt.shape
    # Coordenadas del centro de cada pixel
    lon = transform.c + (np.arange(cols) + 0.5) * transform.a
    lat = transform.f + (np.arange(filas) + 0.5) * transform.e
    print(f"Mapas: {lst.shape[0]} meses de {filas} x {cols} pixeles")
    return lst, ndvi, alt, lat, lon


# ----------------------------------------------------------------------
# Nivelación de cargas
# ----------------------------------------------------------------------
def nivelar(D, n_p):
    """Divide la lista D en n_p bloques casi iguales; los primeros s llevan uno extra."""
    s = len(D) % n_p
    t = (len(D) - s) // n_p
    out = []
    for i in range(n_p):
        if i < s:
            l_i = i * t + i
            l_s = l_i + t + 1
        else:
            l_i = i * t + s
            l_s = l_i + t
        out.append(D[l_i:l_s])
    return out


# ----------------------------------------------------------------------
# Construcción de variables por bloque de filas de la imagen
# ----------------------------------------------------------------------
_MAPAS = {}


def inicializar_mapas(lst, ndvi, alt, lat, lon):
    _MAPAS.update(lst=lst, ndvi=ndvi, alt=alt, lat=lat, lon=lon)


def variables_bloque(filas):
    """Arma una fila por (pixel, mes) para las filas de imagen indicadas."""
    lst, ndvi, alt = _MAPAS["lst"], _MAPAS["ndvi"], _MAPAS["alt"]
    lat, lon = _MAPAS["lat"], _MAPAS["lon"]
    n_meses, _, n_cols = lst.shape
    partes = []
    for r in filas:
        cols = np.arange(n_cols)
        for t in range(max(RETRASOS), n_meses):
            d = {
                "fila": np.full(n_cols, r), "col": cols,
                "anio": np.full(n_cols, ANIO_INICIO + t // 12),
                "mes": np.full(n_cols, t % 12 + 1),
                "LST": lst[t, r],
                "LST_t1": lst[t - 1, r], "LST_t2": lst[t - 2, r],
                "LST_t3": lst[t - 3, r], "LST_t12": lst[t - 12, r],
                "NDVI_t1": ndvi[t - 1, r],
                "ALTITUD": alt[r], "lat": np.full(n_cols, lat[r]), "lon": lon,
            }
            partes.append(pd.DataFrame(d))
    df = pd.concat(partes, ignore_index=True)
    df = df.dropna()                                   # sin nubes ni fuera de la región
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    return df


def construir_variables_v1(mapas, procesos=1):
    """Versión del 2º avance (un DataFrame por fila y mes, se regresa cada bloque
    al proceso principal). Se conserva solo para comparar tiempos.
    Regresa (DataFrame, segundos)."""
    filas = list(range(mapas[2].shape[0]))
    t0 = time.perf_counter()
    if procesos == 1:
        inicializar_mapas(*mapas)
        df = variables_bloque(filas)
    else:
        bloques = nivelar(filas, procesos)
        with mp.Pool(procesos, initializer=inicializar_mapas, initargs=mapas) as pool:
            df = pd.concat(pool.map(variables_bloque, bloques), ignore_index=True)
    return df, time.perf_counter() - t0


# ----------------------------------------------------------------------
# Versión 2: numpy vectorizado + memoria compartida
#
# Problema de la v1: cada proceso recibía una copia de los mapas (≈80 MB) y
# regresaba su bloque de la tabla serializado (pickle); unir los bloques en el
# proceso principal costaba más que armarlos, por eso 8 procesos eran más
# lentos que 1.
#
# En la v2 los mapas y la tabla de salida viven en memoria compartida
# (multiprocessing.shared_memory); los procesos solo reciben índices.
#   Fase 1: cada proceso cuenta cuántos registros válidos tiene su bloque.
#   Suma prefija (proceso principal): posición donde escribe cada bloque.
#   Fase 2: cada proceso escribe sus registros directamente en la tabla.
# Así ningún proceso copia ni regresa datos grandes.
# ----------------------------------------------------------------------
from multiprocessing import shared_memory

COLUMNAS_TABLA = ["fila", "col", "anio", "mes", "LST", "LST_t1", "LST_t2",
                  "LST_t3", "LST_t12", "NDVI_t1", "ALTITUD", "lat", "lon",
                  "mes_sin", "mes_cos"]
_COMP = {}          # arreglos en memoria compartida de cada proceso


def _crear_compartido(arreglo=None, forma=None, tipo="float64"):
    """Copia `arreglo` a memoria compartida, o reserva un arreglo vacío de `forma`."""
    if arreglo is not None:
        forma, tipo = arreglo.shape, arreglo.dtype
    tipo = np.dtype(tipo)
    shm = shared_memory.SharedMemory(create=True, size=max(int(np.prod(forma)) * tipo.itemsize, 1))
    if arreglo is not None:
        np.ndarray(forma, dtype=tipo, buffer=shm.buf)[...] = arreglo
    return shm, (shm.name, tuple(forma), tipo.str)


def _abrir_compartido(desc):
    nombre, forma, tipo = desc
    shm = shared_memory.SharedMemory(name=nombre)
    return shm, np.ndarray(forma, dtype=np.dtype(tipo), buffer=shm.buf)


def _inicializar_v2(desc_lst, desc_ndvi, desc_alt, lat, lon):
    for clave, desc in (("lst", desc_lst), ("ndvi", desc_ndvi), ("alt", desc_alt)):
        shm, arr = _abrir_compartido(desc)
        _COMP[clave] = arr
        _COMP["shm_" + clave] = shm          # se guarda para que no se libere
    _COMP["lat"], _COMP["lon"] = lat, lon


def _mascara_bloque(f0, f1):
    """Registros válidos (sin nubes) del bloque de filas [f0, f1).
    Forma (filas, meses, columnas): el orden es fila -> mes -> columna,
    igual que en la v1, para que la tabla sea idéntica."""
    lst, ndvi, alt = _COMP["lst"][:, f0:f1], _COMP["ndvi"][:, f0:f1], _COMP["alt"][f0:f1]
    r0 = max(RETRASOS)
    ok = np.isfinite(lst[r0:])
    for k in RETRASOS:
        ok &= np.isfinite(lst[r0 - k:lst.shape[0] - k])
    ok &= np.isfinite(ndvi[r0 - 1:-1])
    ok &= np.isfinite(alt)[None]
    return ok.transpose(1, 0, 2)


def _contar_bloque(bloque):
    f0, f1 = bloque
    return int(_mascara_bloque(f0, f1).sum())


def _escribir_bloque(args):
    (f0, f1), inicio, desc_salida = args
    if "salida" not in _COMP:              # el proceso abre la tabla una sola vez
        shm, arr = _abrir_compartido(desc_salida)
        _COMP["salida"], _COMP["shm_salida"] = arr, shm
    lst, ndvi, alt = _COMP["lst"], _COMP["ndvi"], _COMP["alt"]
    ok = _mascara_bloque(f0, f1)
    rr, tt, cc = np.nonzero(ok)            # índices relativos al bloque
    r = rr + f0
    t = tt + max(RETRASOS)                 # mes absoluto (0 = enero 2015)
    mes = t % 12 + 1
    columnas = [r, cc, ANIO_INICIO + t // 12, mes, lst[t, r, cc],
                lst[t - 1, r, cc], lst[t - 2, r, cc], lst[t - 3, r, cc],
                lst[t - 12, r, cc], ndvi[t - 1, r, cc], alt[r, cc],
                _COMP["lat"][r], _COMP["lon"][cc],
                np.sin(2 * np.pi * mes / 12), np.cos(2 * np.pi * mes / 12)]
    salida = _COMP["salida"]
    for j, valores in enumerate(columnas):
        salida[inicio:inicio + len(r), j] = valores
    return len(r)


def construir_variables(mapas, procesos=1):
    """Construye la tabla completa (v2). Regresa (DataFrame, segundos).
    El tiempo incluye crear la memoria compartida y armar el DataFrame."""
    lst, ndvi, alt, lat, lon = mapas
    t0 = time.perf_counter()
    shms = []
    try:
        shm_l, d_l = _crear_compartido(np.ascontiguousarray(lst, dtype="float32"))
        shm_n, d_n = _crear_compartido(np.ascontiguousarray(ndvi, dtype="float32"))
        shm_a, d_a = _crear_compartido(np.ascontiguousarray(alt, dtype="float32"))
        shms += [shm_l, shm_n, shm_a]
        lat64, lon64 = np.asarray(lat, "float64"), np.asarray(lon, "float64")
        bloques = [(b[0], b[-1] + 1) for b in nivelar(list(range(alt.shape[0])), procesos) if b]

        def suma_prefija(conteos):
            inicios = np.concatenate([[0], np.cumsum(conteos)[:-1]]).astype(int)
            return inicios.tolist(), int(sum(conteos))

        if procesos == 1:
            _inicializar_v2(d_l, d_n, d_a, lat64, lon64)
            conteos = [_contar_bloque(b) for b in bloques]              # fase 1
            inicios, n_total = suma_prefija(conteos)
            shm_s, d_s = _crear_compartido(forma=(n_total, len(COLUMNAS_TABLA)))
            shms.append(shm_s)
            escritos = [_escribir_bloque((b, i, d_s)) for b, i in zip(bloques, inicios)]  # fase 2
        else:
            # Un solo Pool para las dos fases (crear procesos en Windows es caro)
            with mp.Pool(procesos, initializer=_inicializar_v2,
                         initargs=(d_l, d_n, d_a, lat64, lon64)) as pool:
                conteos = pool.map(_contar_bloque, bloques, chunksize=1)   # fase 1
                inicios, n_total = suma_prefija(conteos)
                shm_s, d_s = _crear_compartido(forma=(n_total, len(COLUMNAS_TABLA)))
                shms.append(shm_s)
                escritos = pool.map(_escribir_bloque,                      # fase 2
                                    [(b, i, d_s) for b, i in zip(bloques, inicios)], chunksize=1)
        assert escritos == conteos

        tabla = np.ndarray((n_total, len(COLUMNAS_TABLA)), dtype="float64", buffer=shm_s.buf)
        df = pd.DataFrame(tabla.copy(), columns=COLUMNAS_TABLA)
        for c in ("fila", "col", "anio", "mes"):
            df[c] = df[c].astype("int32")
        for c in COLUMNAS_TABLA[4:11]:
            df[c] = df[c].astype("float32")
    finally:
        for clave in [k for k in _COMP if k.startswith("shm_")]:
            _COMP[clave].close()
        _COMP.clear()
        for s in shms:
            s.close()
            s.unlink()
    return df, time.perf_counter() - t0


def particionar(df, anio_prueba=2024, frac_validacion=0.2, n_max_entrenamiento=None, semilla=1):
    """Prueba = el último año (se predicen sus mapas de calor sin haberlos visto).
    Entrenamiento y validación = años anteriores, al azar."""
    prueba = df[df["anio"] == anio_prueba]
    resto = df[df["anio"] < anio_prueba].sample(frac=1, random_state=semilla)
    n_val = int(len(resto) * frac_validacion)
    validacion = resto.iloc[:n_val]
    entrenamiento = resto.iloc[n_val:]
    if n_max_entrenamiento and len(entrenamiento) > n_max_entrenamiento:
        entrenamiento = entrenamiento.iloc[:n_max_entrenamiento]
    if len(validacion) > 50000:
        validacion = validacion.iloc[:50000]
    print(f"Entrenamiento: {len(entrenamiento):,} | Validación: {len(validacion):,} | Prueba ({anio_prueba}): {len(prueba):,}")
    return entrenamiento, validacion, prueba
