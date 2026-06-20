"""
gee_utils.py — Utilitários server-side para Google Earth Engine.
Etapa 3: mask_s2_clouds, build_s2_composite
Etapa 4: add_spectral_indices
Etapa 5: load_mapbiomas
Etapa 10 (melhoria): build_dual_season_composite, DUAL_SEASON_BANDS
"""
import ee

# Bandas-base de uma única estação (10 espectrais + 4 índices).
# Ordem importa: stratifiedSample preserva essa ordem na tabela de amostras.
FEATURE_BANDS = [
    'B2', 'B3', 'B4',          # Azul, Verde, Vermelho (10 m)
    'B5', 'B6', 'B7', 'B8A',   # Red Edge (20 m → GEE faz upsample para 10 m na composição)
    'B8',                       # NIR largo (10 m)
    'B11', 'B12',               # SWIR 1 e 2 (20 m)
    'NDVI', 'NDWI', 'EVI', 'SAVI',
]

# Bandas da composição de dupla estação (Etapa 10):
#   14 da seca (_dry) + 14 da chuva (_wet) + 2 de amplitude sazonal = 30 bandas.
# A amplitude (wet - dry) é a feature mais discriminante para LULC tropical:
#   Agricultura tem amplitude alta (solo nu → soja verde); Solo Exposto ~0; Cerrado intermediária.
DUAL_SEASON_BANDS = (
    [f'{b}_dry' for b in FEATURE_BANDS]
    + [f'{b}_wet' for b in FEATURE_BANDS]
    + ['NDVI_amp', 'EVI_amp']
)


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


def load_mapbiomas(aoi, mb_from, mb_to, year=2023):
    """
    Carrega MapBiomas Collection 9, seleciona o ano, reclassifica para as classes
    do projeto e recorta para a AOI.

    Verifique o asset ID em https://developers.google.com/earth-engine/datasets
    se a coleção estiver indisponível — o MapBiomas atualiza periodicamente.

    Args:
        aoi:      ee.Geometry — área de estudo
        mb_from:  list[int]  — códigos MapBiomas originais (gerado de LULC_CLASSES)
        mb_to:    list[int]  — códigos do projeto correspondentes
        year:     int        — ano a selecionar (default 2023)

    Returns:
        ee.Image com banda 'lulc' contendo valores 1–6 (pixels sem classe = mascarados).
    """
    asset_id = (
        'projects/mapbiomas-public/assets/brazil/lulc/collection9'
        '/mapbiomas_collection90_integration_v1'
    )
    raw = ee.Image(asset_id).select(f'classification_{year}')

    reclassified = (
        raw
        .remap(mb_from, mb_to, defaultValue=0)
        .rename('lulc')
    )
    # Mascara pixels cujo código original não está em mb_from (defaultValue=0)
    return reclassified.updateMask(reclassified.neq(0)).clip(aoi)


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


def _seasonal_composite(aoi, start_date, end_date, suffix, max_cloud_pct):
    """Composite mediana de uma estação, com as 14 bandas-base renomeadas com sufixo."""
    composite = build_s2_composite(aoi, start_date, end_date, max_cloud_pct)
    return composite.rename([f'{b}_{suffix}' for b in FEATURE_BANDS])


def build_dual_season_composite(
    aoi, dry_dates, wet_dates, max_cloud_dry=20, max_cloud_wet=40,
):
    """
    Combina composites de seca e chuva + features de amplitude sazonal (30 bandas).

    A estação chuvosa de MT (~Nov–Mar) tem muito mais nuvem que a seca, por isso
    max_cloud_wet default é mais alto (40%) — a mediana sobre muitas cenas absorve
    os pixels residuais de nuvem. Verifique a contagem de cenas da chuva no notebook.

    Args:
        aoi:           ee.Geometry
        dry_dates:     (start, end) da seca, ex: ('2023-06-01', '2023-09-30')
        wet_dates:     (start, end) da chuva, ex: ('2023-12-01', '2024-03-31')
        max_cloud_dry: limiar de nuvem para a seca
        max_cloud_wet: limiar de nuvem para a chuva (mais alto por padrão)

    Returns:
        ee.Image com as bandas listadas em DUAL_SEASON_BANDS.
    """
    dry = _seasonal_composite(aoi, dry_dates[0], dry_dates[1], 'dry', max_cloud_dry)
    wet = _seasonal_composite(aoi, wet_dates[0], wet_dates[1], 'wet', max_cloud_wet)
    combined = dry.addBands(wet)

    # Amplitude sazonal: diferença chuva − seca dos índices de vegetação
    ndvi_amp = combined.select('NDVI_wet').subtract(combined.select('NDVI_dry')).rename('NDVI_amp')
    evi_amp  = combined.select('EVI_wet').subtract(combined.select('EVI_dry')).rename('EVI_amp')

    return combined.addBands([ndvi_amp, evi_amp]).clip(aoi)
