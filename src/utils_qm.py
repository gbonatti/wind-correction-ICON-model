"""
utils_qm.py
===========
Módulo utilitário central do pipeline ICON + Quantile Mapping.
Coloque este arquivo na pasta src/ do projeto.

Exporta para cada script:
  01_metricas.py      → ANOS, ARQUIVO_*, listar_arquivos_por_estacao,
                         ler_arquivo, aplicar_cv_qm, calcular_linha_metricas
  02_qqplot.py        → ARQUIVO_CORRIGIDO, ESTACOES_DO_ANO
  03_heatmap.py       → ARQUIVO_CORRIGIDO, ESTACOES_DO_ANO, HORAS
  04_relatorio_pdf.py → DIRETORIO_BASE, ANOS, ESTACOES_DO_ANO,
                         mapear_arquivos, carregar_estacao,
                         loocv_qm_mensal, calcular_metricas_completo
  05_mapas_espaciais  → ARQUIVO_CORRIGIDO, ARQUIVO_AGREGADO,
                         ESTACOES_CSV, SHP_PATH
"""

import os
import glob
import numpy as np
import pandas as pd

# ===========================================================================
# CONSTANTES GLOBAIS
# ===========================================================================

# Diretório onde utils_qm.py está salvo (src/).
# Todos os arquivos de entrada/saída são resolvidos relativamente à raiz do
# projeto (um nível acima de src/), independentemente de onde o terminal
# foi aberto.
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)   # raiz do projeto

def _proj(nome):
    """Resolve 'nome' relativo à raiz do projeto."""
    return os.path.join(_PROJECT_DIR, nome)

DIRETORIO_BASE    = _proj('dados')
ANOS              = [2021, 2022, 2023, 2024, 2025]
LIMIAR_MAPE       = 0.5   # m/s — abaixo disso excluído do MAPE
MIN_AMOSTRAS_QM   = 10    # mínimo para calibrar QM em um mês
HORAS             = list(range(24))

ARQUIVO_CORRIGIDO = _proj('dados/dados_corrigidos_cv.csv.gz')
ARQUIVO_DETALHADO = _proj('dados/metricas_detalhadas_v2.csv')
ARQUIVO_AGREGADO  = _proj('dados/metricas_agregadas_v2.csv')
ESTACOES_CSV      = _proj('dados/estacoes.csv')
SHP_PATH          = _proj('shape/SP.shp')

ESTACOES_DO_ANO = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}


# ===========================================================================
# QUANTILE MAPPING
# ===========================================================================

def quantile_mapping(obs_hist, model_hist, model_to_correct, n_quantiles=1001):
    """
    Quantile Mapping empírico.

    Calibra a função de transferência com (obs_hist, model_hist) e aplica
    em model_to_correct — que pode ser um período completamente diferente.

    Remove duplicatas em q_model antes de interpolar para evitar problemas
    com dados de vento que têm muitos zeros (calmaria).
    """
    obs_hist         = np.asarray(obs_hist,         dtype=float)
    model_hist       = np.asarray(model_hist,       dtype=float)
    model_to_correct = np.asarray(model_to_correct, dtype=float)

    if len(obs_hist) < MIN_AMOSTRAS_QM or len(model_hist) < MIN_AMOSTRAS_QM:
        return model_to_correct.copy()   # sem dados suficientes → sem correção

    quantiles = np.linspace(0, 100, n_quantiles)
    q_model   = np.percentile(model_hist, quantiles)
    q_obs     = np.percentile(obs_hist,   quantiles)

    _, idx = np.unique(q_model, return_index=True)   # garante eixo x monotônico
    return np.interp(model_to_correct, q_model[idx], q_obs[idx])


# ===========================================================================
# MÉTRICAS
# ===========================================================================

def kge(obs, sim):
    """
    Kling-Gupta Efficiency  KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
      r = correlação de Pearson
      α = σ_sim / σ_obs   (razão de variabilidade)
      β = μ_sim / μ_obs   (razão de médias / viés)
    Perfeito = 1. Abaixo de −0.41 pior que a média climatológica.
    """
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    m   = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[m], sim[m]
    if len(obs) < 2 or obs.std() == 0 or obs.mean() == 0:
        return np.nan
    r     = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sim.std()  / obs.std())
    beta  = float(sim.mean() / obs.mean())
    return float(1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def metricas_serie(obs, sim, limiar_mape=None):
    """
    Retorna dict com BIAS, RMSE, MAE, MAPE, R, R2, KGE para um par obs/sim.
    """
    if limiar_mape is None:
        limiar_mape = LIMIAR_MAPE

    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    m   = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[m], sim[m]

    nulo = {k: np.nan for k in ['BIAS', 'RMSE', 'MAE', 'MAPE', 'R', 'R2', 'KGE']}
    if len(obs) < 2:
        return nulo

    bias = float((sim - obs).mean())
    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    mae  = float(np.mean(np.abs(sim - obs)))

    mk   = obs > limiar_mape
    mape = float(np.mean(np.abs((obs[mk] - sim[mk]) / obs[mk])) * 100) \
           if mk.sum() > 0 else np.nan

    r  = float(np.corrcoef(obs, sim)[0, 1]) if obs.std() > 0 else np.nan
    r2 = r ** 2 if np.isfinite(r) else np.nan

    return {'BIAS': bias, 'RMSE': rmse, 'MAE': mae,
            'MAPE': mape, 'R': r, 'R2': r2, 'KGE': kge(obs, sim)}


def calcular_linha_metricas(group):
    """
    Calcula todas as métricas (orig + QM + deltas + Skill Score) para um
    grupo pandas (resultado de groupby). Retorna pd.Series.

    Espera colunas: vento_obs, vento_icon, vento_icon_qm.
    """
    obs  = group['vento_obs'].values
    orig = group['vento_icon'].values
    qm   = group['vento_icon_qm'].values

    m_orig = metricas_serie(obs, orig)
    m_qm   = metricas_serie(obs, qm)

    cv = float(obs.std() / obs.mean()) if obs.mean() != 0 else np.nan
    ss = float(1 - m_qm['RMSE'] / m_orig['RMSE']) \
         if m_orig['RMSE'] not in (0, np.nan) and np.isfinite(m_orig['RMSE']) else np.nan

    row = {'amostra': int(len(group))}
    for k, v in m_orig.items():
        row[f'{k}_orig'] = round(v, 4) if np.isfinite(v) else np.nan
    for k, v in m_qm.items():
        row[f'{k}_qm']   = round(v, 4) if np.isfinite(v) else np.nan

    row['SS']         = round(ss, 4) if np.isfinite(ss) else np.nan
    row['SS_RMSE']    = row['SS']   # alias usado em alguns scripts
    ss_mae = float(1 - m_qm['MAE'] / m_orig['MAE']) \
             if m_orig['MAE'] not in (0, np.nan) and np.isfinite(m_orig['MAE']) else np.nan
    row['SS_MAE']     = round(ss_mae, 4) if np.isfinite(ss_mae) else np.nan

    # Aliases maiúsculos — compatibilidade com scripts que usam _QM em vez de _qm
    for metrica in ['BIAS', 'RMSE', 'MAE', 'MAPE', 'R', 'R2', 'KGE']:
        row[f'{metrica}_QM']   = row[f'{metrica}_qm']
        row[f'{metrica}_ORIG'] = row[f'{metrica}_orig']
    row['delta_BIAS'] = round(m_qm['BIAS'] - m_orig['BIAS'], 4)
    row['delta_RMSE'] = round(m_qm['RMSE'] - m_orig['RMSE'], 4)
    row['delta_KGE']  = round(m_qm['KGE']  - m_orig['KGE'],  4) \
                        if np.isfinite(m_qm['KGE']) and np.isfinite(m_orig['KGE']) else np.nan
    row['CV_obs']     = round(cv, 4) if np.isfinite(cv) else np.nan
    return pd.Series(row)


# alias usado por 04_relatorio_pdf.py
calcular_metricas_completo = calcular_linha_metricas


# ===========================================================================
# LEITURA DE ARQUIVOS
# ===========================================================================

def listar_arquivos_por_estacao(diretorio_base=None, anos=None):
    """
    Varre subdiretórios de anos dentro de diretorio_base e retorna:
        { 'A001': ['/.../2021/compara_A001_X.csv', ...], ... }
    """
    if diretorio_base is None:
        diretorio_base = DIRETORIO_BASE
    if anos is None:
        anos = ANOS

    todos = []
    for ano in anos:
        todos.extend(glob.glob(os.path.join(diretorio_base, str(ano), 'compara_*.csv')))

    if not todos:
        raise FileNotFoundError(
            f"Nenhum 'compara_*.csv' encontrado em '{diretorio_base}' "
            f"para os anos {anos}. Verifique o caminho."
        )

    agrupado: dict[str, list[str]] = {}
    for arq in todos:
        eid = os.path.basename(arq).split('_')[1]
        agrupado.setdefault(eid, []).append(arq)
    return agrupado


def ler_arquivo(arq, anos=None):
    """
    Lê um compara_*.csv e adiciona colunas derivadas:
    hora_int, ciclo, mes, ano, estacao.

    Retorna (DataFrame, ano_int) ou (None, None) em caso de erro/vazio.
    """
    if anos is None:
        anos = ANOS
    try:
        partes  = arq.replace('\\', '/').split('/')
        ano_arq = next((int(p) for p in partes if p.isdigit() and int(p) in anos), None)
        if ano_arq is None:
            return None, None

        df = pd.read_csv(arq, sep=';', dtype={'hora': str})
        df['hora']     = df['hora'].str.zfill(4)
        df['hora_int'] = df['hora'].str[:2].astype(int)
        df['ciclo']    = np.where((df['hora_int'] >= 6) & (df['hora_int'] <= 18),
                                  'Diurno', 'Noturno')
        df['data_dt']  = pd.to_datetime(df['data'])
        df['mes']      = df['data_dt'].dt.month
        df['ano']      = ano_arq
        df['estacao']  = os.path.basename(arq).split('_')[1]
        df = df.dropna(subset=['vento_obs', 'vento_icon'])

        if df.empty:
            return None, None
        return df, ano_arq

    except Exception as e:
        print(f"\n  [AVISO] {os.path.basename(arq)}: {e}")
        return None, None


def carregar_estacao(id_est_ou_arquivos, arquivos=None, anos=None):
    """
    Lê e concatena todos os CSVs de uma estação.
    Retorna None se nenhum arquivo carregar com sucesso.

    Aceita duas assinaturas:
        carregar_estacao(lista_arquivos)
        carregar_estacao(id_est, lista_arquivos, anos)
    """
    # compatibilidade com ambas as formas de chamada
    lista = id_est_ou_arquivos if arquivos is None else arquivos
    if anos is None:
        anos = ANOS
    frames = []
    for arq in lista:
        df, _ = ler_arquivo(arq, anos)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


# ===========================================================================
# CROSS-VALIDATION LEAVE-ONE-YEAR-OUT + QM MENSAL
# ===========================================================================

def aplicar_cv_qm(df_estacao, anos=None):
    """
    Executa CV Leave-One-Year-Out com QM mensal estratificado.

    Para cada ano de teste:
      - treino = todos os outros anos
      - para cada mês: calibra QM no treino → aplica no teste

    Retorna DataFrame com 'vento_icon_qm' adicionado (todos os anos),
    ou None se dados insuficientes.
    """
    if anos is None:
        anos = ANOS

    blocos: list[pd.DataFrame] = []

    for ano_teste in anos:
        df_treino = df_estacao[df_estacao['ano'] != ano_teste]
        df_teste  = df_estacao[df_estacao['ano'] == ano_teste].copy()

        if df_treino.empty or df_teste.empty:
            continue

        df_teste['vento_icon_qm'] = df_teste['vento_icon'].copy()

        for mes in range(1, 13):
            tr_m = df_treino[df_treino['mes'] == mes]
            te_m = df_teste [df_teste ['mes'] == mes]
            if te_m.empty:
                continue
            df_teste.loc[te_m.index, 'vento_icon_qm'] = quantile_mapping(
                obs_hist         = tr_m['vento_obs'].values,
                model_hist       = tr_m['vento_icon'].values,
                model_to_correct = te_m['vento_icon'].values,
            )

        blocos.append(df_teste)

    return pd.concat(blocos, ignore_index=True) if blocos else None


# alias usado por 04_relatorio_pdf.py
loocv_qm_mensal = aplicar_cv_qm


# ===========================================================================
# ALIASES — compatibilidade entre todos os scripts
# ===========================================================================
mapear_arquivos = listar_arquivos_por_estacao   # 04_relatorio_pdf.py
ler_estacoes    = listar_arquivos_por_estacao   # nome alternativo
