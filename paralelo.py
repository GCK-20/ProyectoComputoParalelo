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


def construir_variables(mapas, procesos=1):
    """Construye la tabla completa. Regresa (DataFrame, segundos)."""
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
