# LULC Classification — Sentinel-2 + MapBiomas

Pipeline de classificação de uso e cobertura do solo (Land Use and Land Cover) com imagens Sentinel-2,
validado contra o MapBiomas, processado server-side no Google Earth Engine.

Projeto introdutório de uma série em IA aplicada a Sensoriamento Remoto.

## Estrutura

```
LULC_project/
├── environment.yml              # Conda env (fonte da verdade das dependências)
├── requirements.txt             # Espelho pip para uso no Google Colab
├── notebooks/
│   └── 01_pipeline_classificacao.ipynb
├── src/
│   ├── gee_utils.py             # autenticação, mascaramento, índices, MapBiomas
│   ├── sampling.py              # amostragem estratificada e preparo de features
│   └── evaluation.py            # métricas e visualização
└── outputs/
    ├── figuras/
    └── metricas/
```

## Setup local (desenvolvimento)

```bash
conda env create -f environment.yml
conda activate lulc-project

# Para rodar o notebook localmente (kernel do ambiente):
python -m ipykernel install --user --name lulc-project --display-name "Python (lulc-project)"
jupyter notebook notebooks/01_pipeline_classificacao.ipynb
```

## Setup no Google Colab

Na primeira célula do notebook:

```python
!pip install -r requirements.txt -q
```

Seguido de autenticação no GEE:

```python
import ee
ee.Authenticate()
ee.Initialize(project='seu-projeto-gcp')
```

## Dados

| Fonte | Asset GEE | Uso |
|---|---|---|
| Sentinel-2 SR | `COPERNICUS/S2_SR_HARMONIZED` | Imagens de entrada |
| MapBiomas | `projects/mapbiomas-public/...` | Rótulos de referência |

> **Nota:** confirme os asset IDs no [GEE Data Catalog](https://developers.google.com/earth-engine/datasets) antes de usar — o MapBiomas atualiza a coleção periodicamente.

## Decisões metodológicas

_(Documentadas progressivamente — ver notebook e comentários em `src/`)_

- **Área de estudo (AOI):** a definir
- **Período de composição:** a definir
- **Classes LULC:** a definir (simplificação das classes MapBiomas)
- **Mascaramento de nuvem:** banda QA60 (padrão) ou `s2cloudless` (mais preciso, mais lento)
- **Classificador:** Random Forest via GEE (`ee.Classifier.smileRandomForest`) + scikit-learn/XGBoost para comparação
