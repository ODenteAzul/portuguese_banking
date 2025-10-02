# Bank Marketing Campaign - Análise Preditiva

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Repositório de análise de dados do dataset **Bank Marketing (Português)** do Kaggle.  
O **relatório detalhado** está em [technical_report_md.md](technical_report_md.md).

---

## Objetivo
- Priorizar clientes para campanhas de marketing bancário
- Maximizar lucro e ROI
- Garantir modelo deployável e livre de data leakage

---

## Estrutura do Repositório

### Scripts Python
- [`analisador.py`](./analisador.py) - Classe para EDA e análise exploratória  
- [`tratamento.py`](./tratamento.py) - Tratamento de dados e feature engineering  
- [`visualizacoes_novas.py`](./visualizacoes_novas.py) - Visualizações customizadas  
- [`modelling.py`](./modelling.py) - Pipeline de modelagem e avaliação

### Notebooks
- [`exploratory.ipynb`](./exploratory.ipynb) - Notebook de exploração inicial do dataset

### Dados
- [`dados/bank-full.csv`](./dados/bank-full.csv) - Dataset completo utilizado na análise

### Outputs
- `outputs/figures/` - Gráficos gerados  
- `outputs/models/` - Modelos serializados
