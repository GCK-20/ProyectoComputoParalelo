"""
PARTE C: cómputo paralelo.
Búsqueda de hiperparámetros secuencial y en paralelo con multiprocessing
(reparto estático con nivelar() o dinámico con imap_unordered) y métricas
de desempeño: speedup, eficiencia y Karp-Flatt.
"""
import time
import multiprocessing as mp
import numpy as np
import pandas as pd

from datos import nivelar
from modelos import inicializar, evaluar_tarea


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
