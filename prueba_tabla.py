"""
Comprueba que la construcción de la tabla con memoria compartida (v2) produce
exactamente la misma tabla que la versión del 2º avance (v1), con 1, 2 y 4
procesos, y muestra los tiempos.

Uso (desde la carpeta del proyecto, con los .tif junto al notebook):
    python pruebas/prueba_tabla.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import datos


def main():
    mapas = datos.leer_mapas(os.environ.get("RUTA_LST", "LST_mensual.tif"),
                             os.environ.get("RUTA_NDVI", "NDVI_mensual.tif"),
                             os.environ.get("RUTA_ALT", "ALTITUD.tif"))
    v1, t1 = datos.construir_variables_v1(mapas, procesos=1)
    print(f"v1, 1 proceso : {t1:6.2f} s  ({len(v1):,} filas)")
    columnas = list(v1.columns)
    for p in (1, 2, 4):
        v2, t2 = datos.construir_variables(mapas, procesos=p)
        dif = max(float(np.abs(v1[c].to_numpy(float) - v2[c].to_numpy(float)).max()) for c in columnas)
        assert v2[columnas].shape == v1.shape, "las tablas no tienen la misma forma"
        assert dif == 0.0, f"las tablas son diferentes (diferencia máxima {dif})"
        print(f"v2, {p} procesos: {t2:6.2f} s  -> tabla idéntica a la v1")
    print("OK")


if __name__ == "__main__":      # necesario en Windows para usar multiprocessing
    main()
