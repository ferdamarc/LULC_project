# LULC Classification — Sentinel-2 + MapBiomas

Pipeline de classificação de uso e cobertura do solo (Land Use and Land Cover) com imagens Sentinel-2,
validado contra o MapBiomas, processado **server-side** no Google Earth Engine.

Projeto introdutório de uma série em IA aplicada a Sensoriamento Remoto. O objetivo aqui não é
só "classificar pixels": é estabelecer uma base de código limpa e reproduzível (composição de imagens,
amostragem, features espectrais, avaliação) reutilizável nos próximos projetos da série
(séries temporais, detecção de queimadas, segmentação com deep learning).

## Resultado em uma linha

Random Forest com features de **dupla estação** (seca + chuva) atinge **acurácia 0.85 / Kappa 0.82**
em Sapezal (MT), classificando 6 classes LULC contra o MapBiomas Collection 9. Em uma cena nova
(~100 km de distância, sem re-treino) a acurácia cai para **0.75** — gap de generalização de ~7 pontos,
dentro da faixa esperada na literatura (5–15 pts).

## Estrutura

```
LULC_project/
├── environment.yml              # Conda env
├── requirements.txt             # Espelho pip para uso no Google Colab
├── notebooks/
│   └── 01_pipeline_classificacao.ipynb
├── src/
│   ├── gee_utils.py             # mascaramento, índices, MapBiomas, composites
│   ├── sampling.py              # amostragem estratificada e FeatureCollection
│   └── evaluation.py            # métricas (confusão, Kappa, F1) e visualização
└── outputs/
    ├── figuras/                 # PNGs (matriz de confusão, F1, feature importance, comparações)
    └── metricas/                # CSVs de métricas/amostras + modelos serializados (.pkl/.json)
```

## Setup local (desenvolvimento)

```bash
conda env create -f environment.yml
conda activate lulc-project

# Registra o kernel do ambiente para o Jupyter:
python -m ipykernel install --user --name lulc-project --display-name "Python (lulc-project)"
jupyter notebook notebooks/01_pipeline_classificacao.ipynb
```

> O `environment.yml` é a **fonte da verdade** das dependências. Nunca instale pacotes globalmente —
> todo o trabalho acontece dentro do ambiente isolado `lulc-project`.

## Setup no Google Colab

A primeira célula clona o repositório e instala as dependências (o Colab parte de `/content/`,
por isso o clone em vez de caminho relativo). Edite `REPO_BRANCH` para a branch desejada:

```python
REPO_BRANCH = 'dev'
!git clone --branch {REPO_BRANCH} https://github.com/<projeto-git>/LULC_project.git /content/LULC_project
!pip install -r /content/LULC_project/requirements.txt -q
```

Em seguida, autenticação no GEE:

```python
import ee
ee.Authenticate()
ee.Initialize(project='projeto-gcp')        # Colocar seu projeto GCP
```

> **Restrição de hardware:** todo o processamento pesado de imagem roda no GEE (server-side).
> A máquina local / Colab só manipula dados leves (tabelas de amostras, métricas, figuras).
> **Nunca** baixe imagens `.tif` cruas — isso quebra o orçamento de memória e de quota.

## Dados

| Fonte | Asset GEE | Uso |
|---|---|---|
| Sentinel-2 SR | `COPERNICUS/S2_SR_HARMONIZED` | Imagens de entrada (reflectância de superfície) |
| MapBiomas C9 | `projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1` | Rótulos de referência |

> **Nota:** confirme os asset IDs no [GEE Data Catalog](https://developers.google.com/earth-engine/datasets)
> antes de usar — o MapBiomas atualiza a coleção periodicamente.

## Decisões metodológicas

### Área de estudo (AOI)
**Sapezal (MT)** — `ee.Geometry.BBox(-59.20, -13.85, -58.70, -13.35)`, ~3.000 km².
Região de fronteira agrícola no Cerrado, com mosaico claro de floresta, savana, pastagem e
agricultura intensiva (soja) — ideal para ter as 6 classes bem representadas sem AOI gigante
que estoure quota.

### Período de composição
Composição **mediana** sobre o período (a mediana é robusta a nuvens residuais e outliers,
melhor que `mean` para reflectância). Duas estações:
- **Seca:** `2023-06-01` → `2023-09-30` (limiar de nuvem 20%)
- **Chuva:** `2023-12-01` → `2024-03-31` (limiar de nuvem 40%)

O limiar de nuvem da chuva é mais alto **de propósito**: a estação chuvosa de MT tem muito mais
cobertura de nuvem, e a mediana sobre mais cenas absorve os pixels residuais.

### Classes LULC (simplificação do MapBiomas)
Seis classes, reagrupando os códigos do MapBiomas:

| # | Classe | Códigos MapBiomas |
|---|---|---|
| 1 | Floresta | 3 |
| 2 | Cerrado/Savana | 4, 12 |
| 3 | Pastagem | 15 |
| 4 | Agricultura | 18, 21, 39, 40, 41 |
| 5 | Água | 11, 31, 33 |
| 6 | Solo Exposto/Não Vegetado | 24, 25, 30 |

### Features
- **Baseline (14 bandas):** 10 bandas espectrais (B2–B12) + 4 índices (NDVI, NDWI, EVI, SAVI).
- **Modelo final — dupla estação (30 bandas):** 14 da seca (`_dry`) + 14 da chuva (`_wet`) +
  amplitude sazonal (`NDVI_amp`, `EVI_amp` = chuva − seca).

Por que cada índice (ver docstrings em `src/gee_utils.py`): NDVI separa vegetação de solo;
NDWI isola água; EVI não satura em dossel fechado; SAVI reduz o efeito do solo, ajudando a
distinguir Pastagem de Solo Exposto.

### Mascaramento de nuvem
Banda **QA60** (bits 10 = nuvem opaca, 11 = cirros) + escala de DN para reflectância (÷10000).
Alternativa mais precisa porém mais cara: `s2cloudless` (não usada aqui para manter o custo baixo).

### Amostragem
`stratifiedSample` com **200 pontos/classe**, `scale=30` (alinhado à resolução do MapBiomas,
evita pixels mistos). Split treino/teste 80/20 feito **no GEE** (`randomColumn`) para não baixar
dados à toa.

### Classificador
**Random Forest** (100 árvores). Comparamos GEE `smileRandomForest`, scikit-learn RF e XGBoost —
os três convergiram em ~0.82 no baseline, indicando que **o gargalo eram as features, não o
algoritmo**. Daí a aposta na dupla estação (ver abaixo). Sem deep learning por escolha de projeto.

## Resultados

| Modelo | Acurácia | Kappa |
|---|---|---|
| Baseline (estação seca, 14 bandas) | 0.83 | 0.80 |
| **Dupla estação (30 bandas)** | **0.85** | **0.82** |

**Ganhos por classe (dual vs baseline):** Agricultura **+0.11** F1 e Cerrado **+0.09** F1 —
exatamente as classes que mais se confundem em uma estação só (na seca a soja já foi colhida e
"vira" solo nu). Trade-off: Pastagem regrediu ~0.06.

**Descoberta não-óbvia:** a importância de features é dominada pelas bandas da **seca** (`B2_dry`,
`B11_dry`), **não** pela amplitude sazonal (`NDVI_amp`) como se previa. A informação da chuva entra
como **desempate**, não como sinal principal. Bom lembrete de que intuição de RS precisa ser
validada empiricamente.

## Limitações conhecidas

- **Overfitting espacial (~7 pts):** o modelo treinado em Sapezal cai de 0.825 → 0.752 de acurácia
  numa cena a ~100 km (Campo Novo do Parecis), sem re-treino. **Floresta** transfere quase perfeito;
  **Cerrado** (−0.15 F1) e **Solo Exposto** (−0.12 F1) são as piores — suas assinaturas espectrais
  variam mais entre regiões. Para produção, treinar com amostras de múltiplas regiões.
- **Rótulos imperfeitos:** o MapBiomas é referência, não verdade absoluta — tem erro próprio
  (~85–90% de acurácia global). Parte do "erro" do modelo pode ser ruído de rótulo.
- **Um único ano (2023):** sem componente temporal multi-ano; mudanças de cobertura não são
  capturadas. (Foco de um projeto futuro da série.)
- **Custo de memória do GEE:** o composite dual (30 bandas, calculado on-the-fly) estoura memória
  no `stratifiedSample`. **Use `tile_scale=4`** nas chamadas de amostragem do dual — esse é o ajuste
  canônico para o erro `User memory limit exceeded`.
- **`fc_to_dataframe` via `getInfo()`** só é adequado para coleções pequenas (< ~5.000 features).
  Para amostragens maiores, migrar para `Export.table.toDrive()`.

## Reprodutibilidade

Todas as etapas usam `seed=42` (amostragem, split, RF). Rode o notebook de cima a baixo
(**Restart Kernel & Run All**) — as células com `%autoreload` garantem que mudanças em `src/`
sejam refletidas sem reiniciar o kernel manualmente.
