"""
04_relatorio_pdf.py
===================
Gera dois PDFs a partir do dados_corrigidos_cv.csv.gz:

  relatorio_qm_todos_anos.pdf  — LOO-CV completo (todos os anos)
  relatorio_qm_2025.pdf        — somente o ano de avaliação (ANO_AVALIACAO)

Estrutura de cada PDF:
  Pág. 1  — Capa com métricas globais
  Pág. 2  — Mapa espacial: BIAS Original vs BIAS QM (1×2)
  Pág. 3  — Mapa espacial: CV sazonal DJF/MAM/JJA/SON (2×2)
  Pág. 4…N— Uma página por estação:
               · Mini-mapa de localização
               · Q-Q overlay por estação do ano
               · Delta-quantiles por estação do ano
               · Tabela de métricas (Anual + sazonais)
  Pág. N+1— Sumário: ranking por Skill Score e KGE

Depende de:
  utils_qm.py
  dados_corrigidos_cv.csv.gz   (gerado por 01_metricas_v2.py)
  metricas_agregadas_v2.csv    (gerado por 01_metricas_v2.py)
  estacoes.csv                 (colunas: nome;codigo;lat;lon)
  shape/SP.shp                 (shapefile do estado de SP)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, Normalize

import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import geopandas as gpd

warnings.filterwarnings('ignore')

from utils_qm import (
    ANOS, ESTACOES_DO_ANO, ARQUIVO_CORRIGIDO, ARQUIVO_AGREGADO,
    ESTACOES_CSV, SHP_PATH, _PROJECT_DIR,
    calcular_metricas_completo,
)

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
ANO_AVALIACAO = 2025
DPI           = 120
COR_ORIG      = '#E05C2A'
COR_QM        = '#2A7FE0'
COR_REF       = '#888888'
N_QUANTILES   = 100

_DIR_SAIDA   = os.path.join(_PROJECT_DIR, 'graficos')
EXTENT       = [-53.2, -43.9, -25.4, -19.7]
PROJ         = ccrs.PlateCarree()
MARKER_SIZE  = 140

SUBCONJUNTOS = [
    ('todos_anos',      None,           'LOO-CV — Todos os Anos',   'relatorio_qm_todos_anos.pdf'),
    (str(ANO_AVALIACAO), ANO_AVALIACAO, f'Operacional — {ANO_AVALIACAO}', f'relatorio_qm_{ANO_AVALIACAO}.pdf'),
]


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE MAPA
# ---------------------------------------------------------------------------
def _base_mapa(ax, gdf_sp):
    """Aplica fundo branco + shapefile + estilo de borda num eixo cartopy."""
    feat = ShapelyFeature(gdf_sp.geometry, PROJ,
                          facecolor='white', edgecolor='#333333', linewidth=0.8)
    ax.set_extent(EXTENT, crs=PROJ)
    ax.set_facecolor('white')
    ax.add_feature(feat, zorder=2)
    for spine in ax.spines.values():
        spine.set_edgecolor('#AAAAAA')
        spine.set_linewidth(0.7)


def _labels_estacoes(ax, lons, lats, codigos, fontsize=6):
    for lon, lat, cod in zip(lons, lats, codigos):
        ax.text(lon + 0.10, lat + 0.10, str(cod),
                fontsize=fontsize, fontweight='bold', color='#111111',
                transform=PROJ, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])


# ---------------------------------------------------------------------------
# PÁGINA 2 — MAPA DE BIAS ANUAL (1 × 2)
# ---------------------------------------------------------------------------
def pagina_mapa_bias(gdf_sp, metricas, descricao):
    bias_vals = np.concatenate([metricas['BIAS_orig'].values,
                                 metricas['BIAS_qm'].values])
    absmax    = np.ceil(np.nanpercentile(np.abs(bias_vals), 98) * 10) / 10
    norm      = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

    lons    = metricas['lon'].values
    lats    = metricas['lat'].values
    codigos = metricas['estacao'].values

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7.5),
        subplot_kw={'projection': PROJ},
        facecolor='#F7F9FC',
        gridspec_kw={'wspace': 0.04, 'left': 0.01, 'right': 0.88,
                     'top': 0.86, 'bottom': 0.06},
    )

    for ax, vals, subtit in [
        (axes[0], metricas['BIAS_orig'].values, 'ICON Original'),
        (axes[1], metricas['BIAS_qm'].values,   'ICON + Quantile Mapping'),
    ]:
        _base_mapa(ax, gdf_sp)
        sc = ax.scatter(lons, lats, c=vals, cmap='RdBu_r', norm=norm,
                        s=MARKER_SIZE, zorder=5,
                        edgecolors='#222222', linewidths=0.5, transform=PROJ)
        _labels_estacoes(ax, lons, lats, codigos)
        ax.set_title(subtit, fontsize=10, fontweight='bold', pad=5)

    cax = fig.add_axes([0.905, 0.10, 0.009, 0.70])
    cb  = fig.colorbar(sc, cax=cax, extend='both')
    cb.set_label('BIAS (m/s)', fontsize=8.5, labelpad=5)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        f'Distribuição Espacial do BIAS Anual — {descricao}\n'
        'vermelho = superestimativa  |  azul = subestimativa',
        fontsize=12, fontweight='bold', y=0.97,
    )
    return fig


# ---------------------------------------------------------------------------
# PÁGINA 3 — MAPA CV SAZONAL (2 × 2)
# ---------------------------------------------------------------------------
def pagina_mapa_cv(gdf_sp, df_cv, est, descricao):
    df_cv_geo = df_cv.merge(est[['codigo', 'lat', 'lon']],
                            left_on='estacao', right_on='codigo', how='inner')

    cv_global, cv_dados = [], {}
    for sigla, meses in ESTACOES_DO_ANO.items():
        cv_est = (
            df_cv_geo[df_cv_geo['mes'].isin(meses)]
            .groupby(['estacao', 'lat', 'lon'])['vento_icon_qm']
            .agg(lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan)
            .reset_index().rename(columns={'vento_icon_qm': 'cv'}).dropna(subset=['cv'])
        )
        cv_dados[sigla] = cv_est
        cv_global.extend(cv_est['cv'].values.tolist())

    vmax_cv = np.ceil(np.nanpercentile(cv_global, 98) * 100) / 100
    norm_cv = Normalize(vmin=0, vmax=vmax_cv)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 11),
        subplot_kw={'projection': PROJ},
        facecolor='#F7F9FC',
        gridspec_kw={'wspace': 0.04, 'hspace': 0.08,
                     'left': 0.01, 'right': 0.88,
                     'top': 0.90, 'bottom': 0.03},
    )

    sc2 = None
    for ax, sigla in zip(axes.flatten(), list(ESTACOES_DO_ANO.keys())):
        cv_est = cv_dados[sigla]
        _base_mapa(ax, gdf_sp)
        sc2 = ax.scatter(cv_est['lon'].values, cv_est['lat'].values,
                         c=cv_est['cv'].values, cmap='YlOrRd', norm=norm_cv,
                         s=MARKER_SIZE, zorder=5,
                         edgecolors='#222222', linewidths=0.5, transform=PROJ)
        _labels_estacoes(ax, cv_est['lon'].values, cv_est['lat'].values,
                         cv_est['estacao'].values)
        ax.set_title(sigla, fontsize=11, fontweight='bold', pad=5)

    cax = fig.add_axes([0.905, 0.05, 0.018, 0.82])
    cb  = fig.colorbar(sc2, cax=cax, extend='max')
    cb.set_label('CV = σ / μ', fontsize=8.5, labelpad=5)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        f'Coeficiente de Variação Sazonal do Vento — {descricao}',
        fontsize=12, fontweight='bold', y=0.96,
    )
    return fig


# ---------------------------------------------------------------------------
# PÁGINA DE ESTAÇÃO
# ---------------------------------------------------------------------------
def pagina_estacao(id_est, df_full, descricao, gdf_sp, est_row, todas_coords_geo=None):
    percentis = np.linspace(0, N_QUANTILES, N_QUANTILES + 1)

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor('#F7F9FC')

    # GridSpec: linha 0 = QQ+minimap, linha 1 = delta, linha 2 = tabela
    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        height_ratios=[3.2, 2.0, 2.2],
        hspace=0.52, wspace=0.40,
        left=0.05, right=0.98, top=0.91, bottom=0.04
    )

    # ── Mini-mapa de localização (col 4, linhas 0+1) ────────────────────
    ax_map = fig.add_subplot(gs[0:2, 4], projection=PROJ)
    _base_mapa(ax_map, gdf_sp)

    # Todas as estações em cinza
    if est_row is not None and not est_row.empty:
        # Pontos de fundo — todas as estações
        if todas_coords_geo is not None:
            ax_map.scatter(
                todas_coords_geo['lon'].values, todas_coords_geo['lat'].values,
                s=25, c='#AAAAAA', zorder=4, transform=PROJ,
                edgecolors='#666666', linewidths=0.4,
            )
        # Estação atual destacada
        row = est_row.iloc[0]
        ax_map.scatter(
            row['lon'], row['lat'],
            s=80, c=COR_QM, zorder=6, transform=PROJ,
            edgecolors='#111111', linewidths=0.8,
        )
        ax_map.text(
            row['lon'] + 0.15, row['lat'] + 0.15, str(id_est),
            fontsize=7, fontweight='bold', color='#111111',
            transform=PROJ, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground='white')],
        )
    ax_map.set_title('Localização', fontsize=8, fontweight='bold', pad=3)

    # ── Q-Q + Delta-quantiles por estação do ano (4 colunas) ────────────
    for col, (nome_s, meses) in enumerate(ESTACOES_DO_ANO.items()):
        df_s  = df_full[df_full['mes'].isin(meses)]
        ax_qq    = fig.add_subplot(gs[0, col])
        ax_delta = fig.add_subplot(gs[1, col])
        ax_qq.set_facecolor('#FFFFFF')
        ax_delta.set_facecolor('#FFFFFF')

        if df_s.empty or len(df_s) < 10:
            for ax in [ax_qq, ax_delta]:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center',
                        transform=ax.transAxes, color='#AAAAAA', fontsize=8)
            ax_qq.set_title(nome_s, fontsize=9, fontweight='bold')
            continue

        obs    = df_s['vento_obs'].values
        orig   = df_s['vento_icon'].values
        qm_arr = df_s['vento_icon_qm'].values
        q_obs  = np.percentile(obs,    percentis)
        q_orig = np.percentile(orig,   percentis)
        q_qm   = np.percentile(qm_arr, percentis)
        vmax   = max(q_obs.max(), q_orig.max(), q_qm.max()) * 1.05

        # Q-Q
        ax_qq.plot([0, vmax], [0, vmax], color=COR_REF, lw=1.0, ls='--')
        ax_qq.scatter(q_obs, q_orig, c=percentis, cmap='Oranges',
                      s=16, edgecolors='none', alpha=0.7, vmin=0, vmax=100, zorder=3)
        ax_qq.plot(q_obs, q_orig, color=COR_ORIG, lw=1.6, alpha=0.8, zorder=2)
        ax_qq.scatter(q_obs, q_qm, c=percentis, cmap='Blues',
                      s=16, edgecolors='none', alpha=0.7, vmin=0, vmax=100, zorder=4)
        ax_qq.plot(q_obs, q_qm,   color=COR_QM,  lw=1.6, alpha=0.8, zorder=3)
        ax_qq.set_xlim(0, vmax); ax_qq.set_ylim(0, vmax)
        ax_qq.set_aspect('equal', adjustable='box')
        ax_qq.set_xlabel('Obs (m/s)', fontsize=7)
        ax_qq.set_ylabel('Modelo (m/s)', fontsize=7)
        ax_qq.tick_params(labelsize=6.5)
        ax_qq.grid(True, linestyle=':', linewidth=0.5, color='#DDDDDD')
        ax_qq.set_title(nome_s, fontsize=9, fontweight='bold', pad=4)

        rmse_o = np.sqrt(np.mean((q_obs - q_orig) ** 2))
        rmse_q = np.sqrt(np.mean((q_obs - q_qm)   ** 2))
        ax_qq.text(0.04, 0.97,
                   f'RMSE-Q\nOrig:{rmse_o:.3f}\nQM:  {rmse_q:.3f}\nn={len(df_s):,}',
                   transform=ax_qq.transAxes, fontsize=6, va='top',
                   bbox=dict(boxstyle='round,pad=0.3', fc='#F0F4FA',
                             ec='#CCCCCC', alpha=0.92))

        # Delta-quantiles
        ax_delta.axhline(0, color=COR_REF, lw=1.0, ls='--')
        ax_delta.fill_between(percentis, q_orig - q_obs, 0, alpha=0.18, color=COR_ORIG)
        ax_delta.fill_between(percentis, q_qm   - q_obs, 0, alpha=0.18, color=COR_QM)
        ax_delta.plot(percentis, q_orig - q_obs, color=COR_ORIG, lw=1.5)
        ax_delta.plot(percentis, q_qm   - q_obs, color=COR_QM,   lw=1.5)
        ax_delta.set_xlabel('Percentil (%)', fontsize=7)
        ax_delta.set_ylabel('Δ q  (m/s)', fontsize=7)
        ax_delta.tick_params(labelsize=6.5)
        ax_delta.grid(True, linestyle=':', linewidth=0.5, color='#DDDDDD')

    # ── Tabela de métricas ───────────────────────────────────────────────
    ax_tab = fig.add_subplot(gs[2, :])
    ax_tab.axis('off')

    col_labels = ['Período', 'N',
                  'BIAS\nOrig', 'BIAS\nQM',
                  'RMSE\nOrig', 'RMSE\nQM',
                  'MAE\nOrig',  'MAE\nQM',
                  'KGE\nOrig',  'KGE\nQM',
                  'R²\nOrig',   'R²\nQM',
                  'SS\nRMSE',   'SS\nMAE']

    linhas = []
    for nome_p, meses in [('Anual', None)] + list(ESTACOES_DO_ANO.items()):
        df_p = df_full if meses is None else df_full[df_full['mes'].isin(meses)]
        if df_p.empty or len(df_p) < 10:
            continue
        m = calcular_metricas_completo(df_p)
        ss_rmse = m.get('SS_RMSE', float('nan'))
        ss_mae  = m.get('SS_MAE',  float('nan'))
        linhas.append([
            nome_p,              f"{int(m['amostra']):,}",
            f"{m['BIAS_orig']:+.3f}", f"{m['BIAS_QM']:+.3f}",
            f"{m['RMSE_orig']:.3f}",  f"{m['RMSE_QM']:.3f}",
            f"{m['MAE_orig']:.3f}",   f"{m['MAE_QM']:.3f}",
            f"{m['KGE_orig']:.3f}",   f"{m['KGE_QM']:.3f}",
            f"{m['R2_orig']:.3f}",    f"{m['R2_qm']:.3f}",
            f"{ss_rmse:+.3f}" if np.isfinite(ss_rmse) else '—',
            f"{ss_mae:+.3f}"  if np.isfinite(ss_mae)  else '—',
        ])

    if linhas:
        table = ax_tab.table(cellText=linhas, colLabels=col_labels,
                             loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 1.45)

        for j in range(len(col_labels)):
            c = table[(0, j)]
            c.set_facecolor('#1A3A5C')
            c.set_text_props(color='white', fontweight='bold')

        for i, linha in enumerate(linhas, 1):
            # Linha "Anual" com destaque leve
            bg = '#EEF4FF' if i == 1 else ('#F7F9FC' if i % 2 == 0 else '#FFFFFF')
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor(bg)
            # Coloração SS RMSE
            try:
                ss = float(linha[-2])
                table[(i, len(col_labels) - 2)].set_facecolor(
                    '#C8E6C9' if ss > 0 else '#FFCDD2')
            except (ValueError, TypeError):
                pass
            # Coloração BIAS QM (col 3)
            try:
                bqm = float(linha[3])
                borig = float(linha[2])
                if abs(bqm) < abs(borig):
                    table[(i, 3)].set_facecolor('#C8E6C9')
            except (ValueError, TypeError):
                pass

    # Legenda
    fig.legend(
        handles=[
            Line2D([0],[0], color=COR_REF,  lw=1.5, ls='--', label='1:1 / zero ref.'),
            Line2D([0],[0], color=COR_ORIG, lw=2.2,           label='ICON Original'),
            Line2D([0],[0], color=COR_QM,   lw=2.2,           label='ICON + QM'),
        ],
        loc='lower center', ncol=3, fontsize=8.5, frameon=True,
        framealpha=0.9, bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        f'Estação {id_est}   |   {descricao}   |   n = {len(df_full):,}',
        fontsize=12, fontweight='bold', y=0.972, color='#222222',
    )
    return fig


# ---------------------------------------------------------------------------
# PÁGINA DE CAPA
# ---------------------------------------------------------------------------
def pagina_capa(total_estacoes, stats_globais, descricao, anos_usados):
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor('#12213A')

    def txt(x, y, s, **kw):
        fig.text(x, y, s, ha='center', va='center', **kw)

    txt(0.5, 0.83, 'Relatório de Correção de Viés do Vento',
        fontsize=24, fontweight='bold', color='white')
    txt(0.5, 0.75, 'ICON Original  vs  ICON + Quantile Mapping',
        fontsize=16, color='#7AAAD0')
    txt(0.5, 0.67, 'Método: Leave-One-Year-Out CV  |  QM Mensal Estratificado',
        fontsize=12, color='#AABBCC')
    txt(0.5, 0.61, f'{descricao}   |   Anos: {anos_usados}   |   Estações: {total_estacoes}',
        fontsize=11, color='#889AAA')

    boxes = [
        ('SS RMSE médio',   f"{stats_globais.get('SS_RMSE', float('nan')):+.3f}"),
        ('KGE Orig médio',  f"{stats_globais.get('KGE_orig', float('nan')):.3f}"),
        ('KGE QM médio',    f"{stats_globais.get('KGE_QM',  float('nan')):.3f}"),
        ('BIAS Orig médio', f"{stats_globais.get('BIAS_orig', float('nan')):+.3f} m/s"),
        ('BIAS QM médio',   f"{stats_globais.get('BIAS_QM',  float('nan')):+.3f} m/s"),
    ]
    for k, (label, val) in enumerate(boxes):
        x  = 0.15 + k * 0.175
        ax = fig.add_axes([x - 0.065, 0.31, 0.13, 0.17])
        ax.set_facecolor('#1E3355')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#4A6A9A')
        ax.text(0.5, 0.65, val, ha='center', va='center', fontsize=14,
                fontweight='bold', color='#7EDDFF', transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=8,
                color='#AABBCC', transform=ax.transAxes)

    # Índice de páginas
    txt(0.5, 0.20, 'Conteúdo:', fontsize=10, fontweight='bold', color='#AABBCC')
    indice = [
        'Pág. 2 — Mapa espacial: BIAS Anual (Original vs QM)',
        'Pág. 3 — Mapa espacial: Coeficiente de Variação Sazonal',
        f'Pág. 4 a {total_estacoes + 3} — Q-Q plots + tabela de métricas por estação',
        f'Pág. {total_estacoes + 4} — Sumário: ranking por Skill Score e KGE',
    ]
    for i, linha in enumerate(indice):
        txt(0.5, 0.15 - i * 0.048, linha, fontsize=9, color='#778899')

    txt(0.5, 0.04, 'Gerado com utils_qm.py + scripts 01–05',
        fontsize=8, color='#445566')
    return fig


# ---------------------------------------------------------------------------
# PÁGINA DE SUMÁRIO — ranking das estações
# ---------------------------------------------------------------------------
def pagina_sumario(df_metricas, descricao):
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor('#F7F9FC')

    fig.suptitle(f'Sumário — Ranking das Estações  |  {descricao}',
                 fontsize=13, fontweight='bold', y=0.97, color='#1A3A5C')

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12,
                           left=0.04, right=0.97, top=0.91, bottom=0.04)

    for col, (metrica, titulo_col, cor_header) in enumerate([
        ('SS_RMSE', 'Ranking por Skill Score (RMSE)', '#1A5276'),
        ('KGE_QM',  'Ranking por KGE — ICON + QM',    '#145A32'),
    ]):
        ax = fig.add_subplot(gs[col])
        ax.axis('off')

        df_rank = df_metricas[['estacao', 'BIAS_orig', 'BIAS_QM',
                                'RMSE_orig', 'RMSE_QM', 'KGE_orig',
                                'KGE_QM', 'SS_RMSE']].copy()
        df_rank = df_rank.sort_values(metrica, ascending=False).reset_index(drop=True)
        df_rank.insert(0, 'Rank', range(1, len(df_rank) + 1))

        col_labels = ['#', 'Estação',
                      'BIAS\nOrig', 'BIAS\nQM',
                      'RMSE\nOrig', 'RMSE\nQM',
                      'KGE\nOrig',  'KGE\nQM',
                      'SS\nRMSE']

        def fmt(v):
            try: return f'{float(v):+.3f}' if abs(float(v)) < 99 else '—'
            except: return '—'

        rows = []
        for _, r in df_rank.iterrows():
            rows.append([
                int(r['Rank']), str(r['estacao']),
                fmt(r['BIAS_orig']), fmt(r['BIAS_QM']),
                fmt(r['RMSE_orig']), fmt(r['RMSE_QM']),
                fmt(r['KGE_orig']),  fmt(r['KGE_QM']),
                fmt(r['SS_RMSE']),
            ])

        table = ax.table(cellText=rows, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.35)

        for j in range(len(col_labels)):
            c = table[(0, j)]
            c.set_facecolor(cor_header)
            c.set_text_props(color='white', fontweight='bold')

        for i, row in enumerate(rows, 1):
            bg = '#F7F9FC' if i % 2 == 0 else '#FFFFFF'
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor(bg)
            # Destaca SS RMSE
            try:
                ss = float(row[-1])
                table[(i, len(col_labels) - 1)].set_facecolor(
                    '#C8E6C9' if ss > 0 else '#FFCDD2')
            except (ValueError, TypeError):
                pass
            # Top 3 com fundo dourado
            if i <= 3:
                table[(i, 0)].set_facecolor('#FFF9C4')
                table[(i, 0)].set_text_props(fontweight='bold')

        ax.set_title(titulo_col, fontsize=10, fontweight='bold',
                     color=cor_header, pad=8)

    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
print('=' * 60)
print('  Script 04 — Relatório PDF')
print('=' * 60)

# Verifica dependências
for arq, nome in [(ARQUIVO_CORRIGIDO, 'dados_corrigidos_cv.csv.gz'),
                  (ARQUIVO_AGREGADO,  'metricas_agregadas_v2.csv'),
                  (ESTACOES_CSV,      'estacoes.csv'),
                  (SHP_PATH,          'shape/SP.shp')]:
    if not os.path.exists(arq):
        print(f'\n[ERRO] Arquivo não encontrado: {arq}')
        raise SystemExit(1)

print(f'  Carregando dados...')
df_all   = pd.read_csv(ARQUIVO_CORRIGIDO, compression='gzip')
metricas = pd.read_csv(ARQUIVO_AGREGADO, sep=';', decimal=',')
est      = pd.read_csv(ESTACOES_CSV, sep=';')
est.columns   = est.columns.str.strip()
est['codigo'] = est['codigo'].astype(str).str.strip()
gdf_sp   = gpd.read_file(SHP_PATH)

# Junta coordenadas às métricas
metricas['estacao'] = metricas['estacao'].astype(str).str.strip()
metricas_geo = metricas.merge(est[['codigo', 'lat', 'lon', 'nome']],
                               left_on='estacao', right_on='codigo', how='inner')

# Tabela de todas as coords para o mini-mapa
todas_coords = est[['codigo', 'lat', 'lon']].copy()

print(f'  Registros: {len(df_all):,}  |  Estações: {df_all["estacao"].nunique()}\n')

for sufixo, ano_filtro, descricao, nome_pdf in SUBCONJUNTOS:

    df = df_all[df_all['ano'] == ano_filtro] if ano_filtro else df_all

    if df.empty:
        print(f'  [AVISO] Sem dados para "{sufixo}", pulando.\n')
        continue

    estacoes    = sorted(df['estacao'].unique())
    total       = len(estacoes)
    arquivo_pdf = os.path.join(_DIR_SAIDA, nome_pdf)
    anos_usados = [ano_filtro] if ano_filtro else ANOS

    print(f'  [{descricao}]  →  {arquivo_pdf}')

    # Métricas globais para a capa
    metr_global = calcular_metricas_completo(df)

    # Métricas por estação para o sumário (apenas estações com coordenadas)
    df_metr_rank = metricas_geo.copy()

    paginas = 0
    with PdfPages(arquivo_pdf) as pdf:

        # Pág 1 — Capa
        fig = pagina_capa(total, metr_global, descricao, anos_usados)
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # Pág 2 — Mapa BIAS anual
        print(f'    Gerando mapa de BIAS...')
        fig = pagina_mapa_bias(gdf_sp, metricas_geo, descricao)
        pdf.savefig(fig, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # Pág 3 — Mapa CV sazonal
        print(f'    Gerando mapa de CV sazonal...')
        fig = pagina_mapa_cv(gdf_sp, df, est, descricao)
        pdf.savefig(fig, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # Págs 4…N — Uma por estação
        for idx, eid in enumerate(estacoes, 1):
            print(f'    [{idx/total*100:5.1f}%] {eid} ({idx}/{total})', end='\r')
            df_est = df[df['estacao'] == eid]
            if df_est.empty:
                continue

            # Prepara dados para o mini-mapa
            est_match = metricas_geo[metricas_geo['estacao'] == eid]
            est_row_simple = est_match if not est_match.empty else None

            fig = pagina_estacao(eid, df_est, descricao, gdf_sp,
                                 est_row_simple, todas_coords)
            pdf.savefig(fig, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            paginas += 1

        # Pág N+1 — Sumário ranking
        print(f'\n    Gerando sumário...')
        fig = pagina_sumario(df_metr_rank, descricao)
        pdf.savefig(fig, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        info = pdf.infodict()
        info['Title']   = f'Relatório QM — {descricao}'
        info['Subject'] = 'Quantile Mapping | LOO-CV | QM Mensal'

    total_pag = paginas + 4   # capa + mapa bias + mapa cv + sumário
    print(f'    Salvo: {arquivo_pdf}  ({total_pag} páginas)\n')

print('=' * 60)
print('  CONCLUÍDO')
print(f'  graficos/relatorio_qm_todos_anos.pdf  — LOO-CV completo')
print(f'  graficos/relatorio_qm_{ANO_AVALIACAO}.pdf        — somente {ANO_AVALIACAO}')
print('=' * 60)
