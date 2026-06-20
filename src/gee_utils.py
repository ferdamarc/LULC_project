"""
gee_utils.py — Utilitários server-side para Google Earth Engine.
Etapa 3: mask_s2_clouds, build_s2_composite
Etapa 4: add_spectral_indices
Etapa 5: load_mapbiomas  (implementado na Etapa 5)
"""
import ee

# Bandas que compõem o stack de features do classificador.
# Ordem importa: sampleRegions preserva essa ordem na tabela de amostras.
FEATURE_BANDS = [
    'B2', 'B3', 'B4',          # Azul, Verde, Vermelho (10 m)
    'B5', 'B6', 'B7', 'B8A',   # Red Edge (20 m → GEE faz upsample para 10 m na composição)
    'B8',                       # NIR largo (10 m)
    'B11', 'B12',               # SWIR 1 e 2 (20 m)
    'NDVI', 'NDWI', 'EVI', 'SAVI',
]


def mask_s2_clouds(image):
    """
    Mascara nuvens opacas (bit 10) e cirros (bit 11) usando a banda QA60.
    Escala os DNs de Sentinel-2 (0–10 000) para reflectância física (0–1).
    copyProperties preserva metadados de data necessários para séries temporais futuras.
    """
    qa = image.select('QA60')
    cloud_mask  = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
    return (
        image
        .updateMask(cloud_mask.And(cirrus_mask))
        .divide(10000)
        .copyProperties(image, ['system:time_start'])
    )


def add_spectral_indices(image):
    """
    Adiciona NDVI, NDWI, EVI e SAVI como bandas extras.
    Deve ser aplicada após mask_s2_clouds (assume reflectância em [0, 1]).

    Por que esses 4 índices para Sapezal/MT:
      NDVI  → separa vegetação densa de solo exposto/área urbana
      NDWI  → isola corpos d'água (rios Juruena e Papagaio na AOI)
      EVI   → não satura em dossel de floresta fechada onde NDVI ≈ 0.9 para tudo
      SAVI  → reduz influência do solo exposto no sinal — diferencia Pastagem de Solo Exposto
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # NDWI (Gao 1996): usa NIR em vez de SWIR — mais sensível a água superficial
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # EVI com coeficientes padrão (Liu & Huete 1995)
    evi = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)', {
            'NIR':  image.select('B8'),
            'RED':  image.select('B4'),
            'BLUE': image.select('B2'),
        }
    ).rename('EVI')

    # SAVI com fator de ajuste de solo L=0.5 (padrão para cobertura vegetal moderada)
    savi = image.expression(
        '1.5 * (NIR - RED) / (NIR + RED + 0.5)', {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
        }
    ).rename('SAVI')

    return image.addBands([ndvi, ndwi, evi, savi])


def build_s2_composite(aoi, start_date, end_date, max_cloud_pct=20):
    """
    Carrega a coleção COPERNICUS/S2_SR_HARMONIZED, filtra por AOI, período e
    cobertura de nuvem, aplica máscara e índices, e retorna um composite mediana.

    O filtro CLOUDY_PIXEL_PERCENTAGE atua sobre metadados da cena (server-side,
    barato) antes do mascaramento pixel a pixel — reduz custo de processamento.

    Retorna:
        ee.Image com as bandas listadas em FEATURE_BANDS (10 espectrais + 4 índices).
    """
    return (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
        .map(mask_s2_clouds)
        .map(add_spectral_indices)
        .select(FEATURE_BANDS)
        .median()
        .clip(aoi)
    )
