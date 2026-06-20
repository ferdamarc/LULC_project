"""
sampling.py — Amostragem estratificada e conversão FeatureCollection → DataFrame.
Etapa 6 do pipeline LULC.
"""
import ee
import pandas as pd


def stratified_sample(lulc_image, composite, aoi, n_per_class=200, scale=30,
                      seed=42, tile_scale=1):
    """
    Amostra n_per_class pontos por classe LULC e extrai os valores do composite S2.

    Combina lulc + composite em uma única ee.Image antes de chamar stratifiedSample,
    que é a forma mais eficiente no GEE (uma única passagem server-side).
    scale=30 alinha a amostragem à resolução do MapBiomas — evita pixels mistos.

    tile_scale: divisor de tiles para reduzir uso de memória server-side. Aumente
    (2, 4, 8, 16) se o GEE retornar "User memory limit exceeded" — comum em
    composites pesados (ex: dual-season com 30 bandas calculadas on-the-fly).
    Cada dobra de tile_scale reduce a memória por tile ao custo de mais tempo.

    Se uma classe tiver menos pixels disponíveis que n_per_class, o GEE retorna
    quantos encontrar (sem erro). Verifique a contagem real após a chamada.
    """
    combined = composite.addBands(lulc_image)
    return combined.stratifiedSample(
        numPoints=n_per_class,
        classBand='lulc',
        region=aoi,
        scale=scale,
        seed=seed,
        geometries=False,
        tileScale=tile_scale,
    )


def train_test_split(samples, train_ratio=0.8, seed=42):
    """
    Adiciona coluna 'random' uniforme [0,1] e divide em train/test.
    O split é feito no GEE para evitar baixar dados desnecessários.
    """
    samples = samples.randomColumn('random', seed)
    train = samples.filter(ee.Filter.lt('random', train_ratio))
    test  = samples.filter(ee.Filter.gte('random', train_ratio))
    return train, test


def fc_to_dataframe(fc, feature_bands, label_col='lulc'):
    """
    Converte FeatureCollection para pandas DataFrame via getInfo().
    Adequado para coleções pequenas (< 5.000 features).
    Para coleções maiores, use Export.table.toDrive() no lugar.

    Garante a ordem das colunas: features primeiro, rótulo por último.
    """
    props = feature_bands + [label_col]
    data = fc.select(props).getInfo()
    rows = [feat['properties'] for feat in data['features']]
    return pd.DataFrame(rows)[props]
