"""
Evaluación temporal: los 12 mapas de 2024 se predicen sin haberlos visto y se
comparan contra tres referencias sencillas. Un modelo solo es útil si le gana
a las referencias.

  - Persistencia t-1:  el mes que viene será igual al mes anterior.
  - Persistencia t-12: el mes será igual al mismo mes del año anterior.
  - Climatología:      el mes será igual al promedio de ese mes en ese pixel
                       durante los años de entrenamiento (2016-2023).
"""
import numpy as np
import pandas as pd

from modelos import metricas

NOMBRES_MES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def climatologia(df, anio_prueba=2024):
    """Promedio de LST de cada pixel en cada mes con los años anteriores a la prueba."""
    hist = df[df["anio"] < anio_prueba]
    return (hist.groupby(["fila", "col", "mes"])["LST"].mean()
            .rename("pred_Climatologia").reset_index())


def agregar_referencias(prueba, clima):
    """Agrega a la tabla de prueba las predicciones de las tres referencias."""
    prueba = prueba.copy()
    prueba["pred_Persistencia_t1"] = prueba["LST_t1"]
    prueba["pred_Persistencia_t12"] = prueba["LST_t12"]
    prueba = prueba.merge(clima, on=["fila", "col", "mes"], how="left")
    # Pixel-mes sin historia (no pasa con estos datos, pero por si acaso)
    prueba["pred_Climatologia"] = prueba["pred_Climatologia"].fillna(prueba["LST_t12"])
    return prueba


def tabla_metricas(prueba, nombres):
    """RMSE, MAE y R2 en 2024 de cada columna pred_<nombre>."""
    filas = [{"metodo": n, **metricas(prueba["LST"], prueba[f"pred_{n}"])} for n in nombres]
    return pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)


def rmse_por_mes(prueba, nombres):
    filas = []
    for mes, g in prueba.groupby("mes"):
        fila = {"mes": NOMBRES_MES[mes - 1]}
        for n in nombres:
            fila[n] = float(np.sqrt(np.mean((g["LST"] - g[f"pred_{n}"]) ** 2)))
        filas.append(fila)
    return pd.DataFrame(filas)


def a_mapa(prueba, columna, mes, forma):
    """Pasa una columna de la tabla de prueba a una imagen (filas x columnas)."""
    g = prueba[prueba["mes"] == mes]
    mapa = np.full(forma, np.nan, dtype="float32")
    mapa[g["fila"].to_numpy(), g["col"].to_numpy()] = g[columna].to_numpy()
    return mapa
