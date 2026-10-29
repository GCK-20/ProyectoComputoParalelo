
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


COLUMNAS_X = [
    "NDVI", "NDBI", "NDWI", "ALTITUD", "PENDIENTE",
    "lat", "lon", "mes_sin", "mes_cos",
    "urbano", "agricola", "bosque"
]

COLUMNA_Y = "LST"


def limpiar(df):
    df = df.dropna(subset=[
        "LST", "NDVI", "NDBI", "NDWI",
        "ALTITUD", "lat", "lon", "mes"
    ]).copy()

    df = df[(df["LST"] > -20) & (df["LST"] < 75)]

    for c in ["NDVI", "NDBI", "NDWI"]:
        df = df[(df[c] >= -1) & (df[c] <= 1)]

    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    df["urbano"] = (df["COBERTURA"] == 50).astype(int)
    df["agricola"] = (df["COBERTURA"] == 40).astype(int)
    df["bosque"] = (df["COBERTURA"] == 10).astype(int)

    df["PENDIENTE"] = df["PENDIENTE"].fillna(0)

    return df.reset_index(drop=True)


def particionar(
    df,
    anio_prueba=2024,
    frac_validacion=0.2,
    n_max_entrenamiento=None,
    semilla=1
):
    prueba = df[df["anio"] == anio_prueba]

    resto = df[df["anio"] < anio_prueba].sample(
        frac=1,
        random_state=semilla
    )

    n_val = int(len(resto) * frac_validacion)

    validacion = resto.iloc[:n_val]
    entrenamiento = resto.iloc[n_val:]

    if n_max_entrenamiento and len(entrenamiento) > n_max_entrenamiento:
        entrenamiento = entrenamiento.iloc[:n_max_entrenamiento]

    return entrenamiento, validacion, prueba


REJILLA = {
    "KNN": {
        "n_neighbors": [3, 5, 10, 20],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },
    "RF": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 5]
    },
    "SVR": {
        "C": [1, 10, 100],
        "epsilon": [0.1, 0.5],
        "gamma": ["scale", 0.1]
    }
}


def crear_modelo(modelo, params, semilla=1):
    if modelo == "KNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(**params)
        )

    elif modelo == "RF":
        return RandomForestRegressor(
            random_state=semilla,
            **params
        )

    elif modelo == "SVR":
        return make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", **params)
        )

    else:
        raise ValueError(f"Modelo no reconocido: {modelo}")


def metricas(y_real, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "R2": float(r2_score(y_real, y_pred))
    }
