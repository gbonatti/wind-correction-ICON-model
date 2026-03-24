"""
03_heatmap.py
=============
Heatmaps de BIAS horário: anual + 4 sazonais, 2 painéis cada.
Processa vento médio e rajada de vento.

Saídas:
  graficos/heatmaps/<var>/todos_anos/
  graficos/heatmaps/<var>/<ANO_AVALIACAO>/

Depende de: utils_qm.py, dados_corrigidos_cv_<var>.csv.gz
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

from utils_qm import VARIAVEIS, ESTACOES_DO_ANO, HORAS, arquivo_corrigido, _PROJECT_DIR

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
ANO_AVALIACAO = 2025
DPI           = 170


# ---------------------------------------------------------------------------
# MATRIZES E PLOT
# ---------------------------------------------------------------------------
def construir_matrizes(df_sub, cfg, ordem_estacoes=None):
    col_obs   = cfg['col_obs']
    col_model = cfg['col_model']
    col_qm    = cfg['col_qm']

    cols = ['estacao', 'hora_int', col_obs, col_model, col_qm]
    sub  = df_sub[[c for c in cols if c in df_sub.columns]]

    bias_orig = sub[col_model] - sub[col_obs]
    bias_qm   = sub[col_qm]   - sub[col_obs]

    agg = pd.DataFrame({
        'estacao':   sub['estacao'],
        'hora_int':  sub['hora_int'],
        'bias_orig': bias_orig,
        'bias_qm':   bias_qm,
    })

    pivot_orig = agg.pivot_table(
        index='estacao', columns='hora_int', values='bias_orig',
        aggfunc='mean').reindex(columns=HORAS)
    pivot_qm   = agg.pivot_table(
        index='estacao', columns='hora_int', values='bias_qm',
        aggfunc='mean').reindex(columns=HORAS)

    if ordem_estacoes is None:
        ordem = pivot_orig.abs().mean(axis=1).sort_values(ascending=False).index
    else:
        ordem = [e for e in ordem_estacoes if e in pivot_orig.index]

    return pivot_orig.loc[ordem], pivot_qm.loc[ordem], list(ordem)


def plot_heatmap(pivot_orig, pivot_qm, titulo, nome_arquivo, cfg):
    n_est  = len(pivot_orig)
    CELL_H = 0.20
    fig_h  = max(4.0, n_est * CELL_H + 1.6)
    un     = cfg['unidade']

    vmax_bias = np.ceil(max(
        np.nanpercentile(np.abs(pivot_orig.values), 95),
        np.nanpercentile(np.abs(pivot_qm.values),   95),
    ) * 10) / 10
    norm_bias = TwoSlopeNorm(vmin=-vmax_bias, vcenter=0, vmax=vmax_bias)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, fig_h), sharey=True,
                              facecolor='#F7F9FC', constrained_layout=True)

    configs = [
        (axes[0], pivot_orig, 'ICON Original',           '#CC3300'),
        (axes[1], pivot_qm,   'ICON + Quantile Mapping', '#0055AA'),
    ]

    im_last = None
    for ax, pivot, subtit, cor in configs:
        ax.set_facecolor('#DDDDDD')
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                       norm=norm_bias, interpolation='nearest')
        im_last = im
        ax.set_xticks(range(24))
        ax.set_xticklabels([f'{h:02d}h' for h in HORAS],
                           fontsize=6.5, rotation=45, ha='right')
        ax.set_xlabel('Hora UTC', fontsize=8.5, labelpad=4)
        ax.set_yticks(range(n_est))
        ax.set_yticklabels(pivot.index, fontsize=6)
        ax.set_yticks(np.arange(-0.5, n_est, 1), minor=True)
        ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.35)
        ax.tick_params(which='minor', length=0)
        for xc in [5.5, 11.5, 17.5]:
            ax.axvline(xc, color='white', lw=1.0, alpha=0.6)
        ax.set_title(subtit, fontsize=10, fontweight='bold', color=cor, pad=6)

    cb = fig.colorbar(im_last, ax=axes[1], shrink=0.92, pad=0.02, extend='both')
    cb.set_label(f'BIAS ({un})', fontsize=8.5, labelpad=6)
    cb.ax.tick_params(labelsize=7)
    cb.locator = mticker.MaxNLocator(nbins=7, symmetric=True)
    cb.update_ticks()

    for ax in axes:
        for xc, lbl in [(2.5,'Madrugada'),(8.5,'Manhã'),(14.5,'Tarde'),(20.5,'Noite')]:
            ax.text(xc, -1.6, lbl, ha='center', va='top', fontsize=6,
                    color='#666', transform=ax.get_xaxis_transform())

    fig.suptitle(titulo + '\n(vermelho = superestimativa  |  azul = subestimativa)',
                 fontsize=11, fontweight='bold')
    fig.savefig(nome_arquivo, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Salvo: {nome_arquivo}")


# ---------------------------------------------------------------------------
# EXECUÇÃO
# ---------------------------------------------------------------------------
print(f"{'='*60}\n  Heatmap — Vento + Rajada\n{'='*60}")

for var, cfg in VARIAVEIS.items():
    arq_cv = arquivo_corrigido(var)

    if not os.path.exists(arq_cv):
        print(f"\n  [AVISO] Arquivo não encontrado para '{var}': {arq_cv}")
        print(f"  Execute 01_metricas.py primeiro. Pulando '{var}'.\n")
        continue

    print(f"\n  Variável: {cfg['titulo']}")
    print(f"  Carregando {arq_cv}...")
    df = pd.read_csv(arq_cv, compression='gzip')

    subconjuntos = [
        ('todos_anos',       df,                               'LOO-CV — Todos os Anos'),
        (str(ANO_AVALIACAO), df[df['ano'] == ANO_AVALIACAO],  f'Operacional — {ANO_AVALIACAO}'),
    ]

    for sufixo, df_sub, descricao in subconjuntos:
        dir_sub = os.path.join(_PROJECT_DIR, 'graficos', 'heatmaps', var, sufixo)
        os.makedirs(dir_sub, exist_ok=True)

        if df_sub.empty:
            print(f"  Sem dados para '{sufixo}', pulando.")
            continue

        print(f"\n  [{descricao}]  →  {dir_sub}")

        print("    Anual [1/5]...")
        p_orig, p_qm, ordem_global = construir_matrizes(df_sub, cfg)
        plot_heatmap(
            p_orig, p_qm,
            f'BIAS Horário — {cfg["label"]} — {descricao}  |  {len(p_orig)} estações',
            os.path.join(dir_sub, f'heatmap_bias_{var}_anual.png'),
            cfg,
        )

        for i, (sigla, meses) in enumerate(ESTACOES_DO_ANO.items(), 2):
            print(f"    {sigla} [{i}/5]...")
            df_saz = df_sub[df_sub['mes'].isin(meses)]
            if df_saz.empty:
                print(f"    Sem dados para {sigla}, pulando.")
                continue
            p_o, p_q, _ = construir_matrizes(df_saz, cfg, ordem_estacoes=ordem_global)
            plot_heatmap(
                p_o, p_q,
                f'BIAS Horário — {cfg["label"]} — {sigla}  ({descricao})  |  {len(p_o)} estações',
                os.path.join(dir_sub, f'heatmap_bias_{var}_{sigla}.png'),
                cfg,
            )

print(f"\n{'='*60}\n  CONCLUÍDO\n{'='*60}")
