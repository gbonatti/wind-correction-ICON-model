"""
02_qqplot.py
============
Q-Q Plots por estação: principal (3 painéis) + sazonal (4 painéis).
Processa vento médio e rajada de vento.

Saídas:
  graficos/qqplots/<var>/todos_anos/
  graficos/qqplots/<var>/<ANO_AVALIACAO>/

Depende de: utils_qm.py, dados_corrigidos_cv_<var>.csv.gz
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from utils_qm import VARIAVEIS, ESTACOES_DO_ANO, arquivo_corrigido, _PROJECT_DIR

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
N_QUANTILES   = 100
DPI           = 150
ANO_AVALIACAO = 2025
ESTACOES      = []   # [] = todas; ou ['A001', 'A002'] para filtrar

COR_ORIG = '#E05C2A'
COR_QM   = '#2A7FE0'
COR_ZERO = '#888888'


# ---------------------------------------------------------------------------
# AUXILIARES
# ---------------------------------------------------------------------------
def _rmse_q(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _scatter_qq(ax, q_obs, q_model, cor, percentis):
    sc = ax.scatter(q_obs, q_model, c=percentis, cmap='viridis_r',
                    s=22, zorder=3, edgecolors='none', alpha=0.85)
    ax.plot(q_obs, q_model, color=cor, lw=1.4, alpha=0.55, zorder=2)
    return sc

def _anotacao(ax, rmse_q, bias, unidade):
    ax.text(0.04, 0.97,
            f'RMSE-Q = {rmse_q:.3f} {unidade}\nBIAS   = {bias:+.3f} {unidade}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='#EEF2F8', ec='#CCC', alpha=0.9))

def _eixos(ax, vmin, vmax, xlabel, ylabel, titulo, cor_titulo):
    ax.set(xlim=(vmin, vmax), ylim=(vmin, vmax),
           xlabel=xlabel, ylabel=ylabel)
    ax.set_title(titulo, fontweight='bold', fontsize=10, color=cor_titulo, pad=6)
    ax.tick_params(labelsize=8)
    ax.grid(ls=':', lw=0.6, color='#DDD')


# ---------------------------------------------------------------------------
# FIGURAS
# ---------------------------------------------------------------------------
def plot_principal(eid, obs, icon_orig, icon_qm, cfg, dir_saida, var):
    percentis = np.linspace(0, 100, N_QUANTILES + 1)
    q_obs     = np.percentile(obs,       percentis)
    q_orig    = np.percentile(icon_orig, percentis)
    q_qm      = np.percentile(icon_qm,  percentis)

    delta_orig = q_orig - q_obs
    delta_qm   = q_qm  - q_obs
    vmax = max(q_obs.max(), q_orig.max(), q_qm.max()) * 1.05
    un   = cfg['unidade']

    fig = plt.figure(figsize=(16, 5.5), facecolor='#F7F9FC')
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                             left=0.06, right=0.97, top=0.84, bottom=0.13)

    # Painel 1 — Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('white'); ax1.set_aspect('equal', 'box')
    ax1.plot([0, vmax], [0, vmax], color=COR_ZERO, lw=1.2, ls='--', zorder=1)
    sc = _scatter_qq(ax1, q_obs, q_orig, COR_ORIG, percentis)
    fig.colorbar(sc, ax=ax1, fraction=0.035, pad=0.03).set_label('Percentil (%)', fontsize=7.5)
    _anotacao(ax1, _rmse_q(q_obs, q_orig), float((icon_orig - obs).mean()), un)
    _eixos(ax1, 0, vmax, f'Obs. ({un})', f'ICON Original ({un})', 'ICON Original', COR_ORIG)

    # Painel 2 — QM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('white'); ax2.set_aspect('equal', 'box')
    ax2.plot([0, vmax], [0, vmax], color=COR_ZERO, lw=1.2, ls='--', zorder=1)
    sc2 = _scatter_qq(ax2, q_obs, q_qm, COR_QM, percentis)
    fig.colorbar(sc2, ax=ax2, fraction=0.035, pad=0.03).set_label('Percentil (%)', fontsize=7.5)
    _anotacao(ax2, _rmse_q(q_obs, q_qm), float((icon_qm - obs).mean()), un)
    _eixos(ax2, 0, vmax, f'Obs. ({un})', f'ICON + QM ({un})', 'ICON + Quantile Mapping', COR_QM)

    # Painel 3 — Delta-Quantiles
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('white')
    ax3.axhline(0, color=COR_ZERO, lw=1.2, ls='--', zorder=1)
    ax3.plot(percentis, delta_orig, color=COR_ORIG, lw=2.0, label='Original', zorder=3)
    ax3.plot(percentis, delta_qm,   color=COR_QM,   lw=2.0, label='QM',       zorder=3)
    ax3.fill_between(percentis, delta_orig, 0,
                     where=(np.abs(delta_orig) > np.abs(delta_qm)),
                     alpha=0.12, color=COR_ORIG)
    ax3.fill_between(percentis, delta_qm, 0,
                     where=(np.abs(delta_qm) < np.abs(delta_orig)),
                     alpha=0.12, color=COR_QM)
    ax3.set_xlabel('Percentil (%)', fontsize=9)
    ax3.set_ylabel(f'Erro do Quantile ({un})\n[modelo − obs]', fontsize=8.5)
    ax3.set_title('Delta-Quantiles', fontweight='bold', fontsize=10)
    ax3.legend(fontsize=8.5, framealpha=0.9)
    ax3.tick_params(labelsize=8)
    ax3.grid(ls=':', lw=0.6, color='#DDD')

    fig.suptitle(
        f'Q-Q Plot — {cfg["label"]} — Estação {eid}  |  CV LOYO  |  n = {len(obs):,}',
        fontsize=12, fontweight='bold', y=0.97,
    )
    fig.legend(handles=[
        Line2D([0],[0], color=COR_ZERO, lw=1.5, ls='--', label='Linha 1:1'),
        Line2D([0],[0], color=COR_ORIG, lw=2,             label='ICON Original'),
        Line2D([0],[0], color=COR_QM,   lw=2,             label='ICON + QM'),
    ], loc='lower center', ncol=3, fontsize=8.5,
       frameon=True, bbox_to_anchor=(0.5, 0.005))

    out = os.path.join(dir_saida, f'qqplot_{var}_{eid}_principal.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_sazonal(eid, df_est, cfg, dir_saida, var):
    col_obs   = cfg['col_obs']
    col_model = cfg['col_model']
    col_qm    = cfg['col_qm']
    un        = cfg['unidade']
    percentis = np.linspace(0, 100, N_QUANTILES + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='#F7F9FC')
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.38, wspace=0.32, left=0.08, right=0.97, top=0.91, bottom=0.07)

    for i, (sigla, meses) in enumerate(ESTACOES_DO_ANO.items()):
        ax  = axes[i]
        sub = df_est[df_est['mes'].isin(meses)]
        if sub.empty:
            ax.set_visible(False)
            continue

        q_obs  = np.percentile(sub[col_obs].values,   percentis)
        q_orig = np.percentile(sub[col_model].values, percentis)
        q_qm   = np.percentile(sub[col_qm].values,   percentis)
        vmax   = max(q_obs.max(), q_orig.max(), q_qm.max()) * 1.05

        ax.set_facecolor('white'); ax.set_aspect('equal', 'box')
        ax.plot([0, vmax], [0, vmax], color=COR_ZERO, lw=1.0, ls='--', zorder=1)
        ax.scatter(q_obs, q_orig, c=percentis, cmap='Oranges', s=20, zorder=3,
                   edgecolors='none', alpha=0.8)
        ax.plot(q_obs, q_orig, color=COR_ORIG, lw=1.2, alpha=0.5)
        ax.scatter(q_obs, q_qm,  c=percentis, cmap='Blues',   s=20, zorder=4,
                   edgecolors='none', alpha=0.8)
        ax.plot(q_obs, q_qm,  color=COR_QM,   lw=1.2, alpha=0.5)
        ax.set_title(f'{sigla}  (n={len(sub):,})', fontweight='bold', fontsize=10)
        ax.set_xlabel(f'Obs. ({un})', fontsize=8.5)
        ax.set_ylabel(f'Modelo ({un})', fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(ls=':', lw=0.5, color='#DDD')
        ax.set(xlim=(0, vmax), ylim=(0, vmax))
        ax.legend(handles=[
            Line2D([0],[0], color=COR_ORIG, lw=2, label='ICON Original'),
            Line2D([0],[0], color=COR_QM,   lw=2, label='ICON + QM'),
            Line2D([0],[0], color=COR_ZERO, lw=1, ls='--', label='1:1'),
        ], fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        f'Q-Q Sazonal — {cfg["label"]} — Estação {eid}  |  CV LOYO',
        fontsize=12, fontweight='bold', y=0.96,
    )
    out = os.path.join(dir_saida, f'qqplot_{var}_{eid}_sazonal.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# EXECUÇÃO
# ---------------------------------------------------------------------------
print(f"{'='*60}\n  Q-Q Plot — Vento + Rajada\n{'='*60}")

for var, cfg in VARIAVEIS.items():
    arq_cv = arquivo_corrigido(var)

    if not os.path.exists(arq_cv):
        print(f"\n  [AVISO] Arquivo não encontrado para '{var}': {arq_cv}")
        print(f"  Execute 01_metricas.py primeiro. Pulando '{var}'.\n")
        continue

    print(f"\n  Variável: {cfg['titulo']}")
    print(f"  Carregando {arq_cv}...")
    df = pd.read_csv(arq_cv, compression='gzip')

    estacoes = sorted(df['estacao'].unique())
    if ESTACOES:
        estacoes = [e for e in estacoes if e in ESTACOES]
    total = len(estacoes)

    subconjuntos = [
        ('todos_anos',       df,                                'LOO-CV — Todos os Anos'),
        (str(ANO_AVALIACAO), df[df['ano'] == ANO_AVALIACAO],  f'Operacional — {ANO_AVALIACAO}'),
    ]

    total_arq = 0
    for sufixo, df_sub, descricao in subconjuntos:
        dir_sub = os.path.join(_PROJECT_DIR, 'graficos', 'qqplots', var, sufixo)
        os.makedirs(dir_sub, exist_ok=True)
        print(f"  [{descricao}]  →  {dir_sub}")

        df_sub_est = df_sub[df_sub['estacao'].isin(estacoes)]
        if df_sub_est.empty:
            print(f"  Sem dados para este subconjunto, pulando.")
            continue

        for idx, eid in enumerate(estacoes, 1):
            print(f"    [{idx/total*100:5.1f}%] {eid} ({idx}/{total})", end='\r')
            de = df_sub_est[df_sub_est['estacao'] == eid]
            if de.empty:
                continue
            plot_principal(eid,
                           de[cfg['col_obs']].values,
                           de[cfg['col_model']].values,
                           de[cfg['col_qm']].values,
                           cfg, dir_sub, var)
            plot_sazonal(eid, de, cfg, dir_sub, var)
            total_arq += 2
        print()

    print(f"  {total_arq} arquivos gerados para '{var}'.")

print(f"\n{'='*60}\n  CONCLUÍDO\n{'='*60}")
