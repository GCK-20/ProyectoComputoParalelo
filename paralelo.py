"""
Cómputo paralelo con multiprocessing (sin joblib ni n_jobs).

1. Búsqueda de hiperparámetros: cada combinación es una tarea independiente
   (paralelización exploratoria). Reparto estático con nivelar() o dinámico
   con imap_unordered.
2. Predicción de los mapas de 2024: los pixeles a predecir se parten en
   bloques y cada proceso predice los suyos con el mismo modelo
   (paralelismo de datos).
3. Métricas de desempeño: speedup, eficiencia y Karp-Flatt.
"""
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from datos import nivelar
from modelos import inicializar, evaluar_tarea


# ----------------------------------------------------------------------
# 1. Búsqueda de hiperparámetros
# ----------------------------------------------------------------------
def evaluar_bloque(bloque):
    """Reparto estático: un proceso evalúa todas las tareas de su bloque, en orden."""
    return [evaluar_tarea(t) for t in bloque]


def busqueda(tareas, datos, procesos=1, reparto="dinamico"):
    """Evalúa todas las tareas y regresa (resultados, segundos).
    procesos=1 -> secuencial; reparto="estatico" -> bloques fijos con nivelar();
    reparto="dinamico" -> cada proceso toma la siguiente tarea al terminar."""
    t0 = time.perf_counter()
    if procesos == 1:
        inicializar(*datos)
        resultados = [evaluar_tarea(t) for t in tareas]
    else:
        with mp.Pool(processes=procesos, initializer=inicializar, initargs=datos) as pool:
            if reparto == "estatico":
                bloques = nivelar(tareas, procesos)
                resultados = [r for b in pool.map(evaluar_bloque, bloques) for r in b]
            else:
                resultados = list(pool.imap_unordered(evaluar_tarea, tareas, chunksize=1))
    return resultados, time.perf_counter() - t0


# ----------------------------------------------------------------------
# 2. Predicción paralela por bloques de pixeles
# ----------------------------------------------------------------------
_MODELO = {}


def _inicializar_modelo(modelo, X):
    """Cada proceso recibe una sola vez el modelo entrenado y la matriz X."""
    _MODELO["modelo"], _MODELO["X"] = modelo, X


def _predecir_rango(rango):
    i0, i1 = rango
    return i0, _MODELO["modelo"].predict(_MODELO["X"][i0:i1])


def predecir_paralelo(modelo, X, procesos=1, bloques_por_proceso=4):
    """Predice X con `procesos` procesos. Regresa (predicciones, segundos).
    Se hacen varios bloques por proceso para que el reparto dinámico
    compense bloques más lentos."""
    t0 = time.perf_counter()
    if procesos == 1:
        y = modelo.predict(X)
    else:
        n_bloques = procesos * bloques_por_proceso
        indices = nivelar(list(range(len(X))), n_bloques)
        rangos = [(b[0], b[-1] + 1) for b in indices if b]
        y = np.empty(len(X), dtype="float64")
        with mp.Pool(procesos, initializer=_inicializar_modelo, initargs=(modelo, X)) as pool:
            for i0, parte in pool.imap_unordered(_predecir_rango, rangos):
                y[i0:i0 + len(parte)] = parte
    return y, time.perf_counter() - t0


# ----------------------------------------------------------------------
# 3. Métricas de desempeño paralelo
# ----------------------------------------------------------------------
def metricas_paralelas(tiempos):
    """tiempos = {procesos: segundos} -> speedup, eficiencia y Karp-Flatt."""
    t1 = tiempos[1]
    filas = []
    for p, tp in sorted(tiempos.items()):
        s = t1 / tp
        e = (1 / s - 1 / p) / (1 - 1 / p) if p > 1 else np.nan
        filas.append({"procesos": p, "tiempo_s": tp, "speedup": s,
                      "eficiencia": s / p, "karp_flatt": e})
    return pd.DataFrame(filas)
