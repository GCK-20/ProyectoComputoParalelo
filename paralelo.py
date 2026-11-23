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
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)                       # un hilo por proceso
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
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
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
# 2b. KNN con TODOS los datos de entrenamiento (≈3.1 millones de registros)
#
# Enviar el modelo ya entrenado a cada proceso implicaría copiar los 3.1 M de
# registros a cada uno (≈300 MB por proceso). En su lugar, los datos de
# entrenamiento y los pixeles a predecir se ponen en memoria compartida; cada
# proceso construye su propio índice (árbol KD) sobre esos datos, sin
# copiarlos, y predice sus bloques.
# ----------------------------------------------------------------------
from multiprocessing import shared_memory
from threadpoolctl import threadpool_limits
from sklearn.neighbors import KNeighborsRegressor

_KNN = {}


def _a_compartida(arreglo):
    arreglo = np.ascontiguousarray(arreglo, dtype="float64")
    shm = shared_memory.SharedMemory(create=True, size=max(arreglo.nbytes, 1))
    np.ndarray(arreglo.shape, dtype="float64", buffer=shm.buf)[...] = arreglo
    return shm, (shm.name, arreglo.shape)


def _de_compartida(desc):
    nombre, forma = desc
    shm = shared_memory.SharedMemory(name=nombre)
    return shm, np.ndarray(forma, dtype="float64", buffer=shm.buf)


def _inicializar_knn(desc_X, desc_y, desc_P, params):
    threadpool_limits(1)                       # un hilo por proceso
    for clave, desc in (("X", desc_X), ("y", desc_y), ("P", desc_P)):
        shm, arr = _de_compartida(desc)
        _KNN[clave], _KNN["shm_" + clave] = arr, shm
    _KNN["modelo"] = KNeighborsRegressor(**params).fit(_KNN["X"], _KNN["y"])


def _predecir_rango_knn(rango):
    i0, i1 = rango
    return i0, _KNN["modelo"].predict(_KNN["P"][i0:i1])


def predecir_knn_paralelo(X_ent, y_ent, X_pred, params, procesos=1, bloques_por_proceso=4):
    """Entrena KNN con X_ent (ya estandarizado) y predice X_pred.
    Regresa (predicciones, segundos). El tiempo incluye poner los datos en
    memoria compartida, construir el índice y predecir, igual para todo p."""
    t0 = time.perf_counter()
    if procesos == 1:
        with threadpool_limits(1):
            y = KNeighborsRegressor(**params).fit(X_ent, y_ent).predict(X_pred)
        return y, time.perf_counter() - t0
    shms = []
    try:
        for arreglo in (X_ent, y_ent, X_pred):
            shm, desc = _a_compartida(arreglo)
            shms.append((shm, desc))
        descs = [d for _, d in shms]
        indices = nivelar(list(range(len(X_pred))), procesos * bloques_por_proceso)
        rangos = [(b[0], b[-1] + 1) for b in indices if b]
        y = np.empty(len(X_pred), dtype="float64")
        with mp.Pool(procesos, initializer=_inicializar_knn, initargs=(*descs, params)) as pool:
            for i0, parte in pool.imap_unordered(_predecir_rango_knn, rangos):
                y[i0:i0 + len(parte)] = parte
    finally:
        for shm, _ in shms:
            shm.close()
            shm.unlink()
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
