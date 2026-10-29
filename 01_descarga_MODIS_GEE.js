// =====================================================================
//  Descarga de mapas mensuales de temperatura superficial (LST) y NDVI
//  Región: Tlaxcala y Puebla | MODIS Terra, 1 km | 2015–2024 (120 meses)
//
//  CÓMO USARLO
//  1. Entrar a https://code.earthengine.google.com y pegar este código.
//  2. Presionar "Run". En la pestaña "Tasks" aparecen 3 tareas:
//     LST_mensual, NDVI_mensual y ALTITUD. Presionar "RUN" en cada una.
//  3. Se guardan en Google Drive, carpeta "ProyectoParalelo", como GeoTIFF:
//     LST_mensual.tif (120 bandas), NDVI_mensual.tif (120 bandas), ALTITUD.tif
// =====================================================================

var INICIO = ee.Date('2015-01-01');
var MESES = 120;                      // 10 años x 12 meses
var ESCALA = 1000;                    // 1 km por pixel
var CRS = 'EPSG:4326';

var estados = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Mexico'))
  .filter(ee.Filter.inList('ADM1_NAME', ['Tlaxcala', 'Puebla']));
var region = estados.geometry();
print('Estados encontrados (debe ser 2):', estados.size());

var indices = ee.List.sequence(0, MESES - 1);

// Temperatura superficial diurna promedio de cada mes, en °C
// (MOD11A2: compuesto de 8 días; factor de escala 0.02 -> Kelvin)
var lst = ee.ImageCollection(indices.map(function (i) {
  var ini = INICIO.advance(i, 'month');
  return ee.ImageCollection('MODIS/061/MOD11A2')
    .filterDate(ini, ini.advance(1, 'month'))
    .select('LST_Day_1km').mean()
    .multiply(0.02).subtract(273.15)
    .rename('LST');
})).toBands();

// NDVI mensual (MOD13A3; factor de escala 0.0001)
var ndvi = ee.ImageCollection(indices.map(function (i) {
  var ini = INICIO.advance(i, 'month');
  return ee.ImageCollection('MODIS/061/MOD13A3')
    .filterDate(ini, ini.advance(1, 'month'))
    .select('NDVI').mean()
    .multiply(0.0001)
    .rename('NDVI');
})).toBands();

var altitud = ee.Image('USGS/SRTMGL1_003').rename('ALTITUD');

// Pixeles sin dato (fuera de la región o con nubes todo el mes) = -9999
function exportar(imagen, nombre) {
  Export.image.toDrive({
    image: imagen.clip(region).unmask(-9999).toFloat(),
    description: nombre,
    folder: 'ProyectoParalelo',
    region: region.bounds(),
    scale: ESCALA,
    crs: CRS,
    maxPixels: 1e10
  });
}
exportar(lst, 'LST_mensual');
exportar(ndvi, 'NDVI_mensual');
exportar(altitud, 'ALTITUD');

print('Bandas de LST (deben ser 120):', lst.bandNames().size());
Map.centerObject(region, 8);
Map.addLayer(lst.select(3).clip(region), {min: 10, max: 45,
  palette: ['blue', 'cyan', 'yellow', 'orange', 'red']}, 'LST abril 2015 (°C)');
