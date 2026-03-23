"""
03_heatmap_v2.py
================
Heatmaps de BIAS horário: anual + 4 sazonais, 2 painéis cada.
Depende de: utils_qm.py, dados_corrigidos_cv.csv.gz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import os

from utils_qm import ARQUIVO_CORRIGIDO, ESTACOES_DO_ANO, HORAS, _PROJECT_DIR

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
DIR_SAIDA      = os.path.join(_PROJECT_DIR, 'graficos', 'heatmaps')
ANO_AVALIACAO  = 2025  # ano usado no subconjunto operacional
DPI            = 170

os.makedirs(DIR_SAIDA, exist_ok=True)


# ---------------------------------------------------------------------------
# MATRIZES E PLOT
# ---------------------------------------------------------------------------
def construir_matrizes(df_sub, ordem_estacoes=None):
    # Trabalha apenas com as 5 colunas necessárias — evita copiar o DataFrame inteiro
    # e resolve o MemoryError em datasets com milhões de linhas
    cols = ['estacao', 'hora_int', 'vento_obs', 'vento_icon', 'vento_icon_qm']
    sub  = df_sub[cols]   # view, sem cópia

    # Calcula o BIAS diretamente como coluna temporária no subset pequeno
    bias_orig = sub['vento_icon']    - sub['vento_obs']
    bias_qm   = sub['vento_icon_qm'] - sub['vento_obs']

    agg = pd.DataFrame({
        'estacao':    sub['estacao'],
        'hora_int':   sub['hora_int'],
        'bias_orig':  bias_orig,
        'bias_qm':    bias_qm,
    })

    # pivot_table agrega diretamente (média por estação × hora) — sem loop Python
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


def plot_heatmap(pivot_orig, pivot_qm, titulo, nome_arquivo):
    n_est = len(pivot_orig)

    # Altura proporcional ao número de estações, largura fixa
    CELL_H = 0.20   # polegadas por linha de estação
    fig_h  = max(4.0, n_est * CELL_H + 1.6)   # +1.6 para título + eixo x
    fig_w  = 14.0

    vmax_bias = np.ceil(max(
        np.nanpercentile(np.abs(pivot_orig.values), 95),
        np.nanpercentile(np.abs(pivot_qm.values),   95),
    ) * 10) / 10
    norm_bias = TwoSlopeNorm(vmin=-vmax_bias, vcenter=0, vmax=vmax_bias)

    # constrained_layout gerencia espaços automaticamente — sem add_axes manual
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True,
                              facecolor='#F7F9FC',
                              constrained_layout=True)

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

    # Colorbar anexada ao eixo direito — constrained_layout ajusta o espaço
    cb = fig.colorbar(im_last, ax=axes[1], shrink=0.92, pad=0.02, extend='both')
    cb.set_label('BIAS (m/s)', fontsize=8.5, labelpad=6)
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
print(f"{'='*60}\n  Heatmap v2 — Anual + Sazonal\n{'='*60}")
print(f"  Carregando {ARQUIVO_CORRIGIDO}...")

if not os.path.exists(ARQUIVO_CORRIGIDO):
    print(f"\n[ERRO] Arquivo intermediário não encontrado:")
    print(f"  {ARQUIVO_CORRIGIDO}")
    print("\n  Este arquivo é gerado pelo script 01_metricas_v2.py.")
    print("  Certifique-se de estar usando a versão mais recente do")
    print("  01_metricas_v2.py e rode-o antes de executar este script.")
    raise SystemExit(1)

df = pd.read_csv(ARQUIVO_CORRIGIDO, compression='gzip')

subconjuntos = [
    ('todos_anos',       df,                                'LOO-CV — Todos os Anos'),
    (str(ANO_AVALIACAO), df[df['ano'] == ANO_AVALIACAO],   f'Operacional — {ANO_AVALIACAO}'),
]

for sufixo, df_sub, descricao in subconjuntos:
    dir_sub = os.path.join(DIR_SAIDA, sufixo)
    os.makedirs(dir_sub, exist_ok=True)

    if df_sub.empty:
        print(f"  Sem dados para '{sufixo}', pulando.")
        continue

    print(f"\n  [{descricao}]  →  {dir_sub}")

    print("    Anual [1/5]...")
    p_orig, p_qm, ordem_global = construir_matrizes(df_sub)
    plot_heatmap(p_orig, p_qm,
                 f'BIAS Horário — {descricao}  |  {len(p_orig)} estações',
                 os.path.join(dir_sub, 'heatmap_bias_anual.png'))

    for i, (sigla, meses) in enumerate(ESTACOES_DO_ANO.items(), 2):
        print(f"    {sigla} [{i}/5]...")
        df_saz = df_sub[df_sub['mes'].isin(meses)]
        if df_saz.empty:
            print(f"    Sem dados para {sigla}, pulando.")
            continue
        p_o, p_q, _ = construir_matrizes(df_saz, ordem_estacoes=ordem_global)
        plot_heatmap(p_o, p_q,
                     f'BIAS Horário — {sigla}  ({descricao})  |  {len(p_o)} estações',
                     os.path.join(dir_sub, f'heatmap_bias_{sigla}.png'))

print(f"\n{'='*60}")
print(f"  CONCLUÍDO — 10 figuras geradas")
print(f"  graficos/heatmaps/todos_anos/   — LOO-CV completo")
print(f"  graficos/heatmaps/{ANO_AVALIACAO}/      — somente {ANO_AVALIACAO}")
print(f"{'='*60}")
