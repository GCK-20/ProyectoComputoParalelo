"""
Comprueba que la predicción paralela por bloques da el mismo resultado que la
secuencial (con datos sintéticos, no necesita los .tif).

Uso:
    python pruebas/prueba_prediccion.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import modelos
import paralelo as par


def main():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20000, 10))
    y = X[:, 0] * 3 + np.sin(X[:, 1]) + rng.normal(scale=0.1, size=len(X))
    X_nuevo = rng.normal(size=(50000, 10))
    for nombre, params in (("KNN", {"n_neighbors": 10, "weights": "distance", "p": 1}),
                           ("SVR", {"C": 10, "epsilon": 0.1, "gamma": "scale"})):
        m = modelos.crear_modelo(nombre, params).fit(X[:5000], y[:5000])
        y1, t1 = par.predecir_paralelo(m, X_nuevo, procesos=1)
        for p in (2, 4):
            yp, tp = par.predecir_paralelo(m, X_nuevo, procesos=p)
            assert np.allclose(y1, yp), f"{nombre}: la predicción con {p} procesos no coincide"
            print(f"{nombre}: 1 proceso {t1:5.2f} s | {p} procesos {tp:5.2f} s -> mismo resultado")
    print("OK")


if __name__ == "__main__":
    main()
