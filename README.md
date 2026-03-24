# Wind & Gust Correction — ICON Model (São Paulo)

Pipeline de **correção de viés do vento médio e rajada de vento** para o modelo ICON usando **Quantile Mapping (QM)** com **Leave-One-Year-Out Cross-Validation (LOO-CV)** aplicado às estações meteorológicas do estado de São Paulo (2021–2025).

---

## Visão Geral

O modelo ICON frequentemente apresenta erros sistemáticos (viés) na previsão de velocidade do vento e rajadas. Este projeto aplica Quantile Mapping empírico com validação cruzada para:

- Remover o viés de distribuição entre modelo e observações
- Processar **vento médio** e **rajada de vento** de forma independente
- Avaliar a melhora por estação, mês e estação do ano
- Gerar visualizações completas: Q-Q plots, heatmaps horários, mapas espaciais e relatórios PDF

### Método

```
Para cada variável (vento, rajada):
  Para cada ano de teste (LOO-CV):
    treino ← todos os outros anos
    Para cada mês:
      Calibra QM(obs_treino, ICON_treino)
      Aplica QM → ICON_corrigido no ano de teste
```

A estratificação mensal captura a sazonalidade, evitando contaminação entre treino e teste.

---

## Estrutura do Projeto

```
wind-correction-ICON-model/
├── src/
│   ├── utils_qm.py           # Módulo central: QM, métricas, I/O, CV, VARIAVEIS
│   ├── 01_metricas.py        # LOO-CV + QM para vento e rajada → dados intermediários
│   ├── 02_qqplot.py          # Q-Q Plots por estação e variável
│   ├── 03_heatmap.py         # Heatmaps de BIAS horário por variável
│   ├── 04_relatorio_pdf.py   # Relatório PDF por variável
│   └── 05_mapas_espaciais.py # Mapas espaciais de BIAS e CV por variável
│
├── dados/
│   ├── estacoes.csv          # Metadados das estações (nome, código, lat, lon)
│   ├── 2021/                 # compara_XXXX.csv — dados de comparação por ano
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   └── 2025/
│
├── shape/
│   └── SP.shp                # Shapefile do estado de São Paulo
│
├── graficos/                 # Saídas geradas automaticamente
│   ├── qqplots/
│   │   ├── vento/
│   │   └── rajada/
│   ├── heatmaps/
│   │   ├── vento/
│   │   └── rajada/
│   ├── mapas_espaciais/
│   │   ├── vento/
│   │   └── rajada/
│   └── relatorio_qm_<var>_*.pdf
│
├── requirements.txt
└── .gitignore
```

---

## Formato dos Dados de Entrada

Cada arquivo `dados/YYYY/compara_XXXX.csv` deve conter:

```
data;hora;vento_obs;vento_icon;rajada_obs;rajada_icon
2021-01-01;0000;3.5;2.1;6.2;4.8
2021-01-01;0600;4.2;3.8;7.1;5.5
...
```

| Coluna         | Tipo   | Descrição                               |
|----------------|--------|-----------------------------------------|
| `data`         | str    | Data no formato `YYYY-MM-DD`            |
| `hora`         | str    | Hora UTC no formato `HHMM`              |
| `vento_obs`    | float  | Velocidade do vento observada (m/s)     |
| `vento_icon`   | float  | Velocidade do vento prevista pelo ICON  |
| `rajada_obs`   | float  | Rajada de vento observada (m/s)         |
| `rajada_icon`  | float  | Rajada de vento prevista pelo ICON      |

> As colunas de rajada são **opcionais**. Se ausentes em uma estação, o pipeline processa apenas o vento para ela e ignora a rajada sem erros.

O código da estação é extraído do nome do arquivo: `compara_A701_sufixo.csv` → estação `A701`.

---

## Instalação

```bash
git clone https://github.com/gbonatti/wind-correction-ICON-model.git
cd wind-correction-ICON-model
pip install -r requirements.txt
```

> **Dependências:** `numpy`, `pandas`, `matplotlib`, `scipy`, `geopandas`, `cartopy`

---

## Execução

Os scripts devem ser executados na ordem:

```bash
# 1. LOO-CV + QM para vento e rajada → gera todos os dados intermediários
python src/01_metricas.py

# 2. (Independentes — podem rodar em paralelo após o script 01)
python src/02_qqplot.py           # Q-Q Plots (vento + rajada)
python src/03_heatmap.py          # Heatmaps horários (vento + rajada)
python src/04_relatorio_pdf.py    # Relatório PDF (vento + rajada)
python src/05_mapas_espaciais.py  # Mapas espaciais (vento + rajada)
```

### Arquivos intermediários gerados pelo script 01

| Arquivo                                    | Descrição                                      |
|--------------------------------------------|------------------------------------------------|
| `dados/dados_corrigidos_cv_vento.csv.gz`   | Série completa com `vento_icon_qm`             |
| `dados/dados_corrigidos_cv_rajada.csv.gz`  | Série completa com `rajada_icon_qm`            |
| `dados/metricas_agregadas_vento.csv`       | Métricas por estação — vento                   |
| `dados/metricas_agregadas_rajada.csv`      | Métricas por estação — rajada                  |
| `dados/metricas_detalhadas_vento.csv`      | Métricas por estação × mês × ciclo — vento     |
| `dados/metricas_detalhadas_rajada.csv`     | Métricas por estação × mês × ciclo — rajada    |

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

### Q-Q Plots (`graficos/qqplots/<vento|rajada>/`)
- `todos_anos/` — LOO-CV completo
- `2025/` — apenas o ano operacional
- Por estação: `qqplot_<var>_XXXX_principal.png` (3 painéis) + `qqplot_<var>_XXXX_sazonal.png` (4 sazonais)

### Heatmaps (`graficos/heatmaps/<vento|rajada>/`)
- BIAS médio por hora UTC × estação
- Anual + DJF, MAM, JJA, SON
- Escala divergente: vermelho = superestimativa, azul = subestimativa

### Mapas Espaciais (`graficos/mapas_espaciais/<vento|rajada>/`)
- `mapas_bias_anual.png` — BIAS original vs corrigido
- `mapas_cv_sazonal.png` — Coeficiente de Variação por estação do ano

### Relatórios PDF (`graficos/`)
- `relatorio_qm_vento_todos_anos.pdf` / `relatorio_qm_rajada_todos_anos.pdf`
- `relatorio_qm_vento_2025.pdf` / `relatorio_qm_rajada_2025.pdf`

---

## Configuração

Parâmetros globais em `src/utils_qm.py`:

| Constante         | Valor padrão   | Descrição                             |
|-------------------|----------------|---------------------------------------|
| `ANOS`            | `[2021..2025]` | Anos processados                      |
| `MIN_AMOSTRAS_QM` | `10`           | Mínimo de amostras para calibrar o QM |
| `LIMIAR_MAPE`     | `0.5` m/s      | Threshold para cálculo do MAPE        |
| `VARIAVEIS`       | vento, rajada  | Variáveis processadas pelo pipeline   |

Para alterar o ano operacional de avaliação, edite `ANO_AVALIACAO = 2025` nos scripts `02`, `03` e `04`.

Para adicionar ou remover variáveis, edite o dicionário `VARIAVEIS` em `utils_qm.py` — os demais scripts se adaptam automaticamente.

---

## Gilberto Bonatti — Especialista em Modelagem Numérica
