<div align="center">

# Predicción de mapas de calor en Tlaxcala y Puebla
### con búsqueda paralela de hiperparámetros

**Proyecto de Cómputo Paralelo · UPIIT – Instituto Politécnico Nacional**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN%20%7C%20RF%20%7C%20SVR-F7931E?logo=scikitlearn&logoColor=white)
![multiprocessing](https://img.shields.io/badge/Paralelismo-multiprocessing-2a78d6)
![MODIS](https://img.shields.io/badge/Datos-MODIS%201%20km-1baf7a)
![Earth Engine](https://img.shields.io/badge/Google-Earth%20Engine-4285F4?logo=google&logoColor=white)

</div>

---

## Resumen

Predecimos el **mapa de calor (temperatura superficial, LST en °C)** de Tlaxcala y Puebla del mes *t*, pixel por pixel a 1 km, a partir de los mapas de los meses anteriores (*t−1, t−2, t−3, t−12*), el NDVI, la altitud y la ubicación. Usamos **120 mapas mensuales de MODIS (2015–2024)**: millones de registros pixel-mes.

Hay **dos niveles de paralelismo** con `multiprocessing`: la construcción de la tabla (la imagen se parte en bloques de filas) y la búsqueda de hiperparámetros de **KNN, Random Forest y SVR** (46 combinaciones, reparto **estático** vs. **dinámico**). Medimos **tiempo, speedup, eficiencia y Karp-Flatt**.

> El modelo se entrena con 2015–2023 y predice los **12 mapas de 2024**, un año que nunca vio.

## Metodología

<div align="center">
  <img src="figuras/diagrama_metodologia.png" alt="Diagrama de la metodología" width="100%">
</div>

| Paso | Qué se hace |
|:---:|---|
| **1. Datos** | MODIS MOD11A2 (LST) y MOD13A3 (NDVI) mensuales a 1 km y SRTM, descargados con Google Earth Engine |
| **2. Construcción paralela** | Cada proceso arma, para su bloque de filas de la imagen, las variables de cada pixel-mes |
| **3. Partición** | Entrenamiento y validación con 2015–2023 (80 / 20); prueba con los 12 meses de 2024 |
| **4. Búsqueda paralela** | 46 combinaciones de hiperparámetros repartidas entre *p* procesos |
| **5. Selección** | Mejor combinación de cada modelo según el RMSE de validación |
| **6. Evaluación** | Mapas de calor real vs. predicho; RMSE, MAE y R² en 2024; curvas de aprendizaje; speedup y eficiencia |

## Estructura del repositorio

```
ProyectoComputoParalelo
├── 01_descarga_MODIS_GEE.js           # Descarga de los mapas en Google Earth Engine
├── datos.py                           # Lectura de mapas y construcción paralela de la tabla
├── modelos.py                         # KNN, RF, SVR, rejilla y métricas
├── paralelo.py                        # Búsqueda paralela, nivelación y speedup
├── 02_prediccion_mapas_calor.ipynb    # Notebook principal: corre todo el experimento
├── diagrama_metodologia.dot           # Fuente del diagrama de la metodología
├──  figuras/                          # Figuras para el documento (PNG a 200 dpi)
└──  resultados/                       # Tablas en CSV
```

## Cómo reproducirlo

**1. Instalar dependencias**

```bash
pip install numpy pandas scikit-learn matplotlib rasterio jupyter
```

**2. Descargar los datos**

1. Abrir [code.earthengine.google.com](https://code.earthengine.google.com) con una cuenta registrada en Earth Engine.
2. Pegar `01_descarga_MODIS_GEE.js`, presionar **Run** y, en la pestaña **Tasks**, ejecutar las 3 tareas.
3. Descargar `LST_mensual.tif`, `NDVI_mensual.tif` y `ALTITUD.tif` de Google Drive (carpeta `ProyectoParalelo`) y ponerlos junto al notebook.

**3. Correr el experimento**

```bash
jupyter notebook 02_prediccion_mapas_calor.ipynb
```

Ejecutar todas las celdas en orden. Las tablas quedan en `resultados/` y las figuras en `figuras/`.

## Paralelismo

- **Construcción de la tabla:** las filas de la imagen se reparten en bloques con `nivelar()` y cada proceso arma las variables de su bloque (paralelismo de datos).
- **Búsqueda de hiperparámetros:** cada combinación es una **tarea independiente** (paralelización exploratoria).
- Las tareas se reparten con `multiprocessing.Pool`; los datos se envían una sola vez a cada proceso con un inicializador.
- **Reparto estático:** bloques fijos con el algoritmo de nivelación de cargas (`nivelar`).
- **Reparto dinámico:** cada proceso toma la siguiente tarea al terminar (`imap_unordered`), lo que nivela la carga porque un SVR tarda mucho más que un KNN.
- Cada proceso usa un solo hilo de BLAS, y **no se usa `joblib` ni `n_jobs`** de scikit-learn, como pide la materia.

## Resultados

> Se llenan con las salidas del notebook ejecutado con los datos reales.

| Modelo | RMSE (°C) | MAE (°C) | R² |
|---|:---:|:---:|:---:|
| KNN | — | — | — |
| Random Forest | — | — | — |
| SVR | — | — | — |

| Procesos | Tiempo (s) | Speedup | Eficiencia |
|:---:|:---:|:---:|:---:|
| 1 | — | 1.00 | 1.00 |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |

## Equipo

| Integrante | GitHub |
|---|---|
| Dulce Teresa Besares Vasquez | [@dulcetbesares-afk](https://github.com/dulcetbesares-afk) |
| Guadalupe Brised Checa Linarte | [@GCK-20](https://github.com/GCK-20) |
| Brandon Romero Palacios | [@Bran379](https://github.com/Bran379) |

<div align="center">

**Unidad Profesional Interdisciplinaria de Ingeniería Campus Tlaxcala (UPIIT) · IPN**

</div>
