# Wind Correction — ICON Model (São Paulo)

Pipeline de **correção de viés do vento** para o modelo ICON usando **Quantile Mapping (QM)** com **Leave-One-Year-Out Cross-Validation (LOO-CV)** aplicado às estações meteorológicas do estado de São Paulo (2021–2025).

---

## Visão Geral

O modelo ICON frequentemente apresenta erros sistemáticos (viés) na previsão de velocidade do vento. Este projeto aplica Quantile Mapping empírico com validação cruzada para:

- Remover o viés de distribuição entre modelo e observações
- Avaliar a melhora por estação, mês e estação do ano
- Gerar visualizações completas: Q-Q plots, heatmaps horários, mapas espaciais e relatórios PDF

### Método

```
Para cada ano de teste (LOO-CV):
  treino ← todos os outros anos
  Para cada mês:
    Calibra QM(obs_treino, ICON_treino)
    Aplica QM → ICON_corrigido no ano de teste
```

A estratificação mensal captura a sazonalidade do vento, evitando contaminação entre treino e teste.

---

## Estrutura do Projeto

```
wind-correction-ICON-model/
├── src/
│   ├── utils_qm.py          # Módulo central: QM, métricas, I/O, CV
│   ├── 01_metricas.py       # Executa LOO-CV + QM → gera dados intermediários
│   ├── 02_qqplot.py         # Q-Q Plots por estação (principal + sazonal)
│   ├── 03_heatmap.py        # Heatmaps de BIAS horário (anual + 4 sazonais)
│   ├── 04_relatorio_pdf.py  # Relatório PDF completo por estação
│   └── 05_mapas_espaciais.py# Mapas espaciais de BIAS e CV (Cartopy)
│
├── dados/
│   ├── estacoes.csv         # Metadados das estações (nome, código, lat, lon)
│   ├── 2021/                # compara_XXXX.csv — dados de comparação por ano
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   └── 2025/
│
├── shape/
│   └── SP.shp               # Shapefile do estado de São Paulo
│
├── graficos/                # Saídas geradas (criadas automaticamente)
│   ├── qqplots/
│   ├── heatmaps/
│   ├── mapas_espaciais/
│   └── relatorio_qm_*.pdf
│
├── requirements.txt
└── .gitignore
```

---

## Formato dos Dados de Entrada

Cada arquivo `dados/YYYY/compara_XXXX.csv` deve conter:

```
data;hora;vento_obs;vento_icon
2021-01-01;0000;3.5;2.1
2021-01-01;0600;4.2;3.8
...
```

| Coluna       | Tipo   | Descrição                          |
|------------- |--------|------------------------------------|
| `data`       | str    | Data no formato `YYYY-MM-DD`       |
| `hora`       | str    | Hora UTC no formato `HHMM`         |
| `vento_obs`  | float  | Velocidade observada (m/s)         |
| `vento_icon` | float  | Velocidade prevista pelo ICON (m/s)|

O código da estação é extraído do nome do arquivo: `compara_A701_sufixo.csv` → estação `A701`.

---

## Instalação

```bash
git clone https://github.com/SEU_USUARIO/wind-correction-ICON-model.git
cd wind-correction-ICON-model
pip install -r requirements.txt
```

> **Dependências:** `numpy`, `pandas`, `matplotlib`, `scipy`, `geopandas`, `cartopy`

---

## Execução

Os scripts devem ser executados na ordem:

```bash
# 1. Processa LOO-CV + QM → gera dados intermediários e tabelas de métricas
python src/01_metricas.py

# 2. (Independentes — podem rodar em paralelo após o script 01)
python src/02_qqplot.py          # Q-Q Plots
python src/03_heatmap.py         # Heatmaps horários
python src/04_relatorio_pdf.py   # Relatório PDF
python src/05_mapas_espaciais.py # Mapas espaciais
```

### Arquivos intermediários gerados pelo script 01

| Arquivo                               | Descrição                                    |
|---------------------------------------|----------------------------------------------|
| `dados/dados_corrigidos_cv.csv.gz`    | Série completa com coluna `vento_icon_qm`    |
| `dados/metricas_agregadas_v2.csv`     | Métricas por estação (uma linha por estação) |
| `dados/metricas_detalhadas_v2.csv`    | Métricas por estação × mês × ciclo           |

---

## Métricas Calculadas

| Métrica    | Fórmula                                   | Perfeito |
|------------|-------------------------------------------|----------|
| BIAS       | mean(modelo − obs)                        | 0        |
| RMSE       | √mean((modelo − obs)²)                   | 0        |
| MAE        | mean(\|modelo − obs\|)                   | 0        |
| KGE        | 1 − √((r−1)²+(α−1)²+(β−1)²)             | 1        |
| R²         | r²                                        | 1        |
| SS\_RMSE   | 1 − RMSE\_qm / RMSE\_orig                | > 0      |

O **Skill Score (SS\_RMSE)** mede a melhora do QM em relação ao ICON original:
- **SS > 0**: QM melhorou o desempenho
- **SS = 0**: nenhuma melhora
- **SS < 0**: QM piorou (raramente ocorre)

---

## Saídas

### Q-Q Plots (`graficos/qqplots/`)
- `todos_anos/` — LOO-CV completo
- `2025/` — apenas o ano operacional
- Por estação: `qqplot_XXXX_principal.png` (3 painéis) + `qqplot_XXXX_sazonal.png` (4 painéis)

### Heatmaps (`graficos/heatmaps/`)
- BIAS médio por hora UTC × estação
- Anual + DJF, MAM, JJA, SON
- Escala divergente: vermelho = superestimativa, azul = subestimativa

### Mapas Espaciais (`graficos/mapas_espaciais/`)
- `mapas_bias_anual.png` — BIAS original vs corrigido
- `mapas_cv_sazonal.png` — Coeficiente de Variação por estação do ano

### Relatórios PDF (`graficos/`)
- `relatorio_qm_todos_anos.pdf` — análise completa LOO-CV
- `relatorio_qm_2025.pdf` — somente o ano operacional

---

## Configuração

Parâmetros globais em `src/utils_qm.py`:

| Constante        | Valor padrão         | Descrição                                |
|------------------|----------------------|------------------------------------------|
| `ANOS`           | `[2021..2025]`       | Anos processados                         |
| `MIN_AMOSTRAS_QM`| `10`                 | Mínimo de amostras para calibrar o QM    |
| `LIMIAR_MAPE`    | `0.5` m/s            | Threshold para cálculo do MAPE           |

Para alterar o ano operacional de avaliação, edite `ANO_AVALIACAO = 2025` nos scripts `02`, `03` e `04`.

---

## Licença

MIT
