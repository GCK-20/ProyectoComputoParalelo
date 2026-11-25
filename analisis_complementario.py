"""
Análisis complementario pedido en la revisión del 2º avance.

1. Estacionariedad de la serie de tiempo de LST (pruebas ADF y KPSS) antes y
   después de quitar la estacionalidad anual.
2. Curvas de aprendizaje con la partición temporal: error de entrenamiento y
   de validación según el tamaño de la muestra (justifica usar 40,000).
   Cada par (modelo, tamaño) es una tarea independiente y se reparte entre
   procesos con multiprocessing (reparto dinámico).
3. Figuras de desempeño paralelo sin la línea de speedup ideal.

Uso (desde la carpeta del repositorio, con los .tif):
    python analisis_complementario.py
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import ast
import time
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The test statistic is outside")   # KPSS: p > 0.1

import datos
import modelos

RUTA_LST = os.environ.get("RUTA_LST", "LST_mensual.tif")
RUTA_NDVI = os.environ.get("RUTA_NDVI", "NDVI_mensual.tif")
RUTA_ALT = os.environ.get("RUTA_ALT", "ALTITUD.tif")
TAMANOS = [2500, 5000, 10000, 20000, 40000]
TAMANOS_KNN_EXTRA = [160000, 640000, None]          # None = todo el entrenamiento
N_VALIDACION = 50000
COLORES = {"KNN": "#2a78d6", "RF": "#eb6834", "SVR": "#1baf7a"}
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


# ----------------------------------------------------------------------
# 1. Estacionariedad
# ----------------------------------------------------------------------
def pruebas(serie):
    adf = adfuller(serie, autolag="AIC")
    kp = kpss(serie, regression="c", nlags="auto")
    return {"ADF_estadistico": adf[0], "ADF_p": adf[1], "KPSS_estadistico": kp[0], "KPSS_p": kp[1]}


def estacionariedad(lst):
    meses = pd.date_range(f"{datos.ANIO_INICIO}-01-01", periods=lst.shape[0], freq="MS")
    serie = pd.Series(np.nanmean(lst.reshape(lst.shape[0], -1), axis=1), index=meses)
    clima = serie.groupby(serie.index.month).transform("mean")      # ciclo anual promedio
    anomalia = serie - clima
    filas = [{"serie": "LST promedio de la región", **pruebas(serie)},
             {"serie": "Anomalía (sin ciclo anual)", **pruebas(anomalia)}]
    tabla = pd.DataFrame(filas)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(serie.index, serie, color="#eb6834", lw=1.6, label="LST promedio")
    axes[0].plot(serie.index, clima, color="#555555", lw=1, ls="--", label="Ciclo anual promedio")
    axes[0].set_ylabel("°C"); axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("LST promedio mensual de la región (2015–2024)")
    axes[1].plot(anomalia.index, anomalia, color="#2a78d6", lw=1.4)
    axes[1].axhline(0, color="#999999", lw=1)
    axes[1].set_ylabel("°C"); axes[1].set_title("Anomalía: LST menos el promedio de su mes")
    fig.tight_layout(); fig.savefig("figuras/serie_estacionariedad.png", dpi=200); plt.close(fig)
    return tabla


# ----------------------------------------------------------------------
# 2. Curvas de aprendizaje (tareas en paralelo)
# ----------------------------------------------------------------------
_D = {}


def _inicializar(X, y, Xv, yv, mejores):
    _D.update(X=X, y=y, Xv=Xv, yv=yv, mejores=mejores)


def _tarea(args):
    nombre, n = args
    X, y = (_D["X"], _D["y"]) if n is None else (_D["X"][:n], _D["y"][:n])
    t0 = time.perf_counter()
    m = modelos.crear_modelo(nombre, _D["mejores"][nombre]).fit(X, y)
    sub = min(len(X), 20000)                       # error de entrenamiento sobre 20,000 filas
    rmse_ent = modelos.metricas(y[:sub], m.predict(X[:sub]))["RMSE"]
    rmse_val = modelos.metricas(_D["yv"], m.predict(_D["Xv"]))["RMSE"]
    return {"modelo": nombre, "n": len(X), "RMSE_entrenamiento": rmse_ent,
            "RMSE_validacion": rmse_val, "segundos": time.perf_counter() - t0}


def curvas(df, mejores, procesos):
    ent, val, _ = datos.particionar(df, anio_prueba=2024, n_max_entrenamiento=None)
    X = ent[datos.COLUMNAS_X].to_numpy(); y = ent[datos.COLUMNA_Y].to_numpy()
    val = val.sample(n=min(N_VALIDACION, len(val)), random_state=modelos.SEMILLA)
    Xv = val[datos.COLUMNAS_X].to_numpy(); yv = val[datos.COLUMNA_Y].to_numpy()
    # particionar() ya revuelve el entrenamiento: los primeros n son una muestra al azar
    tareas = [(m, n) for m in ("SVR", "RF", "KNN") for n in TAMANOS]
    tareas += [("KNN", n) for n in TAMANOS_KNN_EXTRA]
    with mp.Pool(procesos, initializer=_inicializar, initargs=(X, y, Xv, yv, mejores)) as pool:
        filas = list(pool.imap_unordered(_tarea, tareas))
    tabla = pd.DataFrame(filas).sort_values(["modelo", "n"]).reset_index(drop=True)

    graficar_curvas(tabla)
    return tabla


def _etiqueta(n):
    return f"{n / 1e6:.2g}M" if n >= 1e6 else f"{n / 1e3:g}k"


def graficar_curvas(tabla):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, nombre in zip(axes[:3], ("KNN", "RF", "SVR")):
        g = tabla[tabla["modelo"] == nombre]
        ax.plot(g["n"], g["RMSE_validacion"], marker="o", lw=2, color=COLORES[nombre], label="Validación")
        ax.plot(g["n"], g["RMSE_entrenamiento"], marker="o", lw=1.4, ls="--", color=COLORES[nombre],
                alpha=0.7, label="Entrenamiento")
        ax.axvline(40000, color="#999999", lw=1, ls=":")
        ax.set_xscale("log"); ax.minorticks_off()
        ax.set_xticks(list(g["n"])); ax.set_xticklabels([_etiqueta(n) for n in g["n"]], fontsize=8)
        ax.set_title(nombre); ax.set_xlabel("Registros de entrenamiento"); ax.set_ylim(-0.05, 2.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("RMSE (°C)")
    for nombre in ("KNN", "RF", "SVR"):
        g = tabla[(tabla["modelo"] == nombre) & (tabla["n"] <= 40000)]
        axes[3].plot(g["n"], g["segundos"], marker="o", lw=2, color=COLORES[nombre], label=nombre)
    axes[3].set_xscale("log"); axes[3].set_yscale("log"); axes[3].minorticks_off()
    axes[3].set_xticks(TAMANOS); axes[3].set_xticklabels([_etiqueta(n) for n in TAMANOS], fontsize=8)
    axes[3].set_title("Tiempo de entrenamiento y validación")
    axes[3].set_xlabel("Registros de entrenamiento"); axes[3].set_ylabel("Segundos")
    axes[3].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("figuras/curvas_aprendizaje.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------
# 3. Figuras de desempeño sin speedup ideal
# ----------------------------------------------------------------------
def _tres_paneles(grupos, archivo, titulo):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for etiqueta, t, color in grupos:
        axes[0].plot(t["procesos"], t["tiempo_s"], marker="o", lw=2, color=color, label=etiqueta)
        axes[1].plot(t["procesos"], t["speedup"], marker="o", lw=2, color=color, label=etiqueta)
        axes[2].plot(t["procesos"], t["eficiencia"], marker="o", lw=2, color=color, label=etiqueta)
    for ax, nombre in zip(axes, ("Tiempo (s)", "Speedup", "Eficiencia")):
        ax.set_title(f"{titulo}: {nombre.lower()}" if nombre != "Tiempo (s)" else f"{titulo}: tiempo")
        ax.set_xlabel("Número de procesos"); ax.set_xticks(sorted(grupos[0][1]["procesos"]))
        ax.legend(frameon=False)
    axes[0].set_ylabel("Segundos"); axes[2].set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(archivo, dpi=200); plt.close(fig)


def figuras_sin_ideal():
    c = pd.read_csv("resultados/construccion_tabla_v1_vs_v2.csv")
    _tres_paneles([("v1 (2º avance)", c[c.version.str.startswith("v1")], "#9a9a9a"),
                   ("v2 (memoria compartida)", c[c.version.str.startswith("v2")], "#2a78d6")],
                  "figuras/construccion_tabla_v1_vs_v2.png", "Construcción")
    p = pd.read_csv("resultados/prediccion_paralela.csv")
    _tres_paneles([("KNN con todos los datos", p[p.modelo == "KNN_todos"], "#0b3d91"),
                   ("SVR", p[p.modelo == "SVR"], "#1baf7a")],
                  "figuras/prediccion_paralela.png", "Predicción")
    b = pd.read_csv("resultados/desempeno_paralelo.csv")
    _tres_paneles([("Reparto dinámico", b, "#2a78d6")], "figuras/desempeno_paralelo.png", "Búsqueda")


def main():
    os.makedirs("figuras", exist_ok=True); os.makedirs("resultados", exist_ok=True)
    figuras_sin_ideal()
    print("Figuras de desempeño regeneradas sin speedup ideal")

    mapas = datos.leer_mapas(RUTA_LST, RUTA_NDVI, RUTA_ALT)
    est = estacionariedad(mapas[0])
    est.to_csv("resultados/estacionariedad.csv", index=False)
    print(est.round(4).to_string(index=False))

    df, _ = datos.construir_variables(mapas, procesos=1)
    b = pd.read_csv("resultados/busqueda_hiperparametros.csv")
    mejores = {m: ast.literal_eval(p) for m, p in
               b.sort_values("RMSE").groupby("modelo").head(1)[["modelo", "params"]].values}
    t0 = time.perf_counter()
    cur = curvas(df, mejores, procesos=mp.cpu_count())
    cur.to_csv("resultados/curvas_aprendizaje.csv", index=False)
    print(cur.round(3).to_string(index=False))
    print(f"Curvas de aprendizaje: {time.perf_counter() - t0:.1f} s con {mp.cpu_count()} procesos")


if __name__ == "__main__":
    main()
