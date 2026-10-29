import os

# Cada proceso usa 1 solo hilo de BLAS/OpenMP, para que el único paralelismo
# sea el nuestro (si no, numpy o scikit-learn podrían usar varios núcleos solos)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Variables de entrada del modelo (X) y variable a predecir (y)
COLUMNAS_X = ["NDVI", "NDBI", "NDWI", "ALTITUD", "PENDIENTE", "lat", "lon",
              "mes_sin", "mes_cos", "urbano", "agricola", "bosque"]
COLUMNA_Y = "LST"


# ----------------------------------------------------------------------
# 1. Preparación de datos
# ----------------------------------------------------------------------
def limpiar(df):
    """Quita filas vacías o con valores imposibles y crea variables nuevas."""
    n0 = len(df)
    df = df.dropna(subset=["LST", "NDVI", "NDBI", "NDWI", "ALTITUD", "lat", "lon", "mes"]).copy()
    df = df[(df["LST"] > -20) & (df["LST"] < 75)]            # °C posibles en superficie
    for c in ["NDVI", "NDBI", "NDWI"]:
        df = df[(df[c] >= -1) & (df[c] <= 1)]                 # índices normalizados
    # El mes es cíclico (diciembre está junto a enero): se codifica con seno y coseno
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    # Cobertura ESA WorldCover: 10 árboles, 40 cultivo, 50 construido
    df["urbano"] = (df["COBERTURA"] == 50).astype(int)
    df["agricola"] = (df["COBERTURA"] == 40).astype(int)
    df["bosque"] = (df["COBERTURA"] == 10).astype(int)
    df["PENDIENTE"] = df["PENDIENTE"].fillna(0)
    print(f"Filas: {n0:,} -> {len(df):,} después de limpiar")
    return df.reset_index(drop=True)


def particionar(df, anio_prueba=2024, frac_validacion=0.2, n_max_entrenamiento=None, semilla=1):
    """Prueba = el año más reciente (se predice un año que el modelo no vio).
    Entrenamiento y validación = años anteriores, separados al azar."""
    prueba = df[df["anio"] == anio_prueba]
    resto = df[df["anio"] < anio_prueba].sample(frac=1, random_state=semilla)
    n_val = int(len(resto) * frac_validacion)
    validacion = resto.iloc[:n_val]
    entrenamiento = resto.iloc[n_val:]
    if n_max_entrenamiento and len(entrenamiento) > n_max_entrenamiento:
        entrenamiento = entrenamiento.iloc[:n_max_entrenamiento]
    print(f"Entrenamiento: {len(entrenamiento):,} | Validación: {len(validacion):,} | Prueba ({anio_prueba}): {len(prueba):,}")
    return entrenamiento, validacion, prueba

