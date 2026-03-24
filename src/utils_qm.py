"""
utils_qm.py
===========
Módulo utilitário central do pipeline ICON + Quantile Mapping.

Suporta múltiplas variáveis (vento e rajada) de forma genérica.
"""

import os
import glob
import numpy as np
import pandas as pd

# ===========================================================================
# CAMINHOS
# ===========================================================================
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

def _proj(nome):
    return os.path.join(_PROJECT_DIR, nome)

DIRETORIO_BASE  = _proj('dados')
ANOS            = [2021, 2022, 2023, 2024, 2025]
LIMIAR_MAPE     = 0.5
MIN_AMOSTRAS_QM = 10
HORAS           = list(range(24))
ESTACOES_CSV    = _proj('dados/estacoes.csv')
SHP_PATH        = _proj('shape/SP.shp')

ESTACOES_DO_ANO = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}

# ===========================================================================
# VARIÁVEIS SUPORTADAS
# ===========================================================================
VARIAVEIS = {
    'vento': {
        'col_obs':   'vento_obs',
        'col_model': 'vento_icon',
        'col_qm':    'vento_icon_qm',
        'label':     'Vento',
        'unidade':   'm/s',
        'titulo':    'Vento (m/s)',
    },
    'rajada': {
        'col_obs':   'rajada_obs',
        'col_model': 'rajada_icon',
        'col_qm':    'rajada_icon_qm',
        'label':     'Rajada',
        'unidade':   'm/s',
        'titulo':    'Rajada de Vento (m/s)',
    },
}


# ---------------------------------------------------------------------------
# Caminhos de saída por variável
# ---------------------------------------------------------------------------
def arquivo_corrigido(var='vento'):
    return _proj(f'dados/dados_corrigidos_cv_{var}.csv.gz')

def arquivo_detalhado(var='vento'):
    return _proj(f'dados/metricas_detalhadas_{var}.csv')

def arquivo_agregado(var='vento'):
    return _proj(f'dados/metricas_agregadas_{var}.csv')

# Backward-compat (aponta para 'vento')
ARQUIVO_CORRIGIDO = arquivo_corrigido('vento')
ARQUIVO_DETALHADO = arquivo_detalhado('vento')
ARQUIVO_AGREGADO  = arquivo_agregado('vento')


# ===========================================================================
# QUANTILE MAPPING
# ===========================================================================
def quantile_mapping(obs_hist, model_hist, model_to_correct, n_quantiles=1001):
    """
    Quantile Mapping empírico.
    Calibra com (obs_hist, model_hist) e aplica em model_to_correct.
    Remove duplicatas em q_model para garantir interpolação monotônica.
    """
    obs_hist         = np.asarray(obs_hist,         dtype=float)
    model_hist       = np.asarray(model_hist,       dtype=float)
    model_to_correct = np.asarray(model_to_correct, dtype=float)

    if len(obs_hist) < MIN_AMOSTRAS_QM or len(model_hist) < MIN_AMOSTRAS_QM:
        return model_to_correct.copy()

    quantiles = np.linspace(0, 100, n_quantiles)
    q_model   = np.percentile(model_hist, quantiles)
    q_obs     = np.percentile(obs_hist,   quantiles)

    _, idx = np.unique(q_model, return_index=True)
    return np.interp(model_to_correct, q_model[idx], q_obs[idx])


# ===========================================================================
# MÉTRICAS
# ===========================================================================
def kge(obs, sim):
    """KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²). Perfeito = 1."""
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
    """Retorna dict com BIAS, RMSE, MAE, MAPE, R, R2, KGE."""
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


def calcular_linha_metricas(group, var='vento'):
    """
    Calcula métricas (orig + QM + deltas + Skill Score) para um grupo pandas.
    Parâmetro 'var' seleciona as colunas corretas (vento ou rajada).
    """
    cfg  = VARIAVEIS[var]
    obs  = group[cfg['col_obs']].values
    orig = group[cfg['col_model']].values
    qm   = group[cfg['col_qm']].values

    m_orig = metricas_serie(obs, orig)
    m_qm   = metricas_serie(obs, qm)

    cv = float(obs.std() / obs.mean()) \
         if np.isfinite(obs.mean()) and obs.mean() != 0 else np.nan
    ss = float(1 - m_qm['RMSE'] / m_orig['RMSE']) \
         if m_orig['RMSE'] not in (0, np.nan) and np.isfinite(m_orig['RMSE']) else np.nan

    row = {'amostra': int(len(group))}
    for k, v in m_orig.items():
        row[f'{k}_orig'] = round(v, 4) if np.isfinite(v) else np.nan
    for k, v in m_qm.items():
        row[f'{k}_qm']   = round(v, 4) if np.isfinite(v) else np.nan

    row['SS']      = round(ss, 4) if np.isfinite(ss) else np.nan
    row['SS_RMSE'] = row['SS']
    ss_mae = float(1 - m_qm['MAE'] / m_orig['MAE']) \
             if m_orig['MAE'] not in (0, np.nan) and np.isfinite(m_orig['MAE']) else np.nan
    row['SS_MAE'] = round(ss_mae, 4) if np.isfinite(ss_mae) else np.nan

    for metrica in ['BIAS', 'RMSE', 'MAE', 'MAPE', 'R', 'R2', 'KGE']:
        row[f'{metrica}_QM']   = row[f'{metrica}_qm']
        row[f'{metrica}_ORIG'] = row[f'{metrica}_orig']
    row['delta_BIAS'] = round(m_qm['BIAS'] - m_orig['BIAS'], 4)
    row['delta_RMSE'] = round(m_qm['RMSE'] - m_orig['RMSE'], 4)
    row['delta_KGE']  = round(m_qm['KGE']  - m_orig['KGE'],  4) \
                        if np.isfinite(m_qm['KGE']) and np.isfinite(m_orig['KGE']) else np.nan
    row['CV_obs']     = round(cv, 4) if np.isfinite(cv) else np.nan
    return pd.Series(row)

calcular_metricas_completo = calcular_linha_metricas


# ===========================================================================
# LEITURA DE ARQUIVOS
# ===========================================================================
def listar_arquivos_por_estacao(diretorio_base=None, anos=None):
    """
    Varre subdiretórios de anos e retorna { 'A001': [arquivos...], ... }.
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
    Lê um compara_*.csv e adiciona colunas derivadas.
    Remove linhas onde todas as colunas de observação existentes são NaN.
    Retorna (DataFrame, ano_int) ou (None, None).
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

        # Remove apenas linhas onde TODAS as colunas obs existentes são NaN
        obs_cols = [cfg['col_obs'] for cfg in VARIAVEIS.values()
                    if cfg['col_obs'] in df.columns]
        if obs_cols:
            df = df.dropna(subset=obs_cols, how='all')

        if df.empty:
            return None, None
        return df, ano_arq

    except Exception as e:
        print(f"\n  [AVISO] {os.path.basename(arq)}: {e}")
        return None, None


def carregar_estacao(id_est_ou_arquivos, arquivos=None, anos=None):
    """Lê e concatena todos os CSVs de uma estação."""
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
def aplicar_cv_qm(df_estacao, var='vento', anos=None):
    """
    CV Leave-One-Year-Out com QM mensal estratificado para a variável especificada.

    Para cada ano de teste:
      - treino = todos os outros anos
      - para cada mês: calibra QM no treino → aplica no teste

    Retorna DataFrame com coluna QM adicionada, ou None se dados insuficientes.
    """
    if anos is None:
        anos = ANOS

    cfg       = VARIAVEIS[var]
    col_obs   = cfg['col_obs']
    col_model = cfg['col_model']
    col_qm    = cfg['col_qm']

    if col_obs not in df_estacao.columns or col_model not in df_estacao.columns:
        return None

    df = df_estacao.dropna(subset=[col_obs, col_model]).copy()
    if df.empty:
        return None

    blocos: list[pd.DataFrame] = []
    for ano_teste in anos:
        df_treino = df[df['ano'] != ano_teste]
        df_teste  = df[df['ano'] == ano_teste].copy()
        if df_treino.empty or df_teste.empty:
            continue

        df_teste[col_qm] = df_teste[col_model].copy()
        for mes in range(1, 13):
            tr_m = df_treino[df_treino['mes'] == mes]
            te_m = df_teste [df_teste ['mes'] == mes]
            if te_m.empty:
                continue
            df_teste.loc[te_m.index, col_qm] = quantile_mapping(
                obs_hist         = tr_m[col_obs].values,
                model_hist       = tr_m[col_model].values,
                model_to_correct = te_m[col_model].values,
            )
        blocos.append(df_teste)

    return pd.concat(blocos, ignore_index=True) if blocos else None

loocv_qm_mensal = aplicar_cv_qm

# ===========================================================================
# ALIASES
# ===========================================================================
mapear_arquivos = listar_arquivos_por_estacao
ler_estacoes    = listar_arquivos_por_estacao
