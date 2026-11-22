"""
Modelos de regresión (KNN, Random Forest y SVR), la rejilla de
hiperparámetros y las métricas de error.

Cada combinación de la rejilla es una tarea independiente: un proceso la
entrena con los datos de entrenamiento y la evalúa con los de validación.
Así la búsqueda se puede repartir entre procesos (ver paralelo.py).
"""
import os
import time
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEMILLA = 1

# 16 + 18 + 12 = 46 combinaciones
REJILLA = {
    "KNN": {"n_neighbors": [3, 5, 10, 20],
            "weights": ["uniform", "distance"],
            "p": [1, 2]},
    "RF": {"n_estimators": [50, 100, 200],
           "max_depth": [10, 20, None],
           "min_samples_leaf": [1, 5]},
    "SVR": {"C": [1, 10, 100],
            "epsilon": [0.1, 0.5],
            "gamma": ["scale", 0.1]},
}


def crear_modelo(modelo, params, semilla=SEMILLA):
    """Regresa el modelo sin entrenar. KNN y SVR usan variables estandarizadas.
    No se usa n_jobs: todo el paralelismo se hace con multiprocessing."""
    if modelo == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(**params))
    if modelo == "RF":
        return RandomForestRegressor(random_state=semilla, **params)
    if modelo == "SVR":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", **params))
    raise ValueError(f"Modelo no reconocido: {modelo}")


def metricas(y_real, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
            "MAE": float(mean_absolute_error(y_real, y_pred)),
            "R2": float(r2_score(y_real, y_pred))}


def lista_de_tareas(rejilla=REJILLA):
    """Una tarea = (modelo, diccionario de hiperparámetros)."""
    tareas = []
    for modelo, valores in rejilla.items():
        nombres = list(valores)
        for combinacion in product(*valores.values()):
            tareas.append((modelo, dict(zip(nombres, combinacion))))
    return tareas


# Datos de cada proceso: se cargan una sola vez con el initializer del Pool
_DATOS = {}


def inicializar(X_ent, y_ent, X_val, y_val):
    _DATOS.update(X_ent=X_ent, y_ent=y_ent, X_val=X_val, y_val=y_val)


def evaluar_tarea(tarea):
    """Entrena una combinación y la evalúa con validación."""
    modelo, params = tarea
    t0 = time.perf_counter()
    m = crear_modelo(modelo, params)
    m.fit(_DATOS["X_ent"], _DATOS["y_ent"])
    r = metricas(_DATOS["y_val"], m.predict(_DATOS["X_val"]))
    return {"modelo": modelo, "params": str(params), **r,
            "segundos": time.perf_counter() - t0, "pid": os.getpid()}
