"""
05_mapas_espaciais.py
=====================
Gera mapas de distribuição espacial para vento médio e rajada de vento.

Por variável (vento / rajada):
  Mapa 1 — BIAS anual: ICON Original vs ICON + QM  (1 × 2)
  Mapa 2 — CV sazonal do modelo corrigido  (2 × 2)

Saídas:
  graficos/mapas_espaciais/<var>/mapas_bias_anual.png
  graficos/mapas_espaciais/<var>/mapas_cv_sazonal.png

Depende de:
  utils_qm.py
  metricas_agregadas_<var>.csv   (gerado por 01_metricas.py)
  dados_corrigidos_cv_<var>.csv.gz
  estacoes.csv
  shape/SP.shp
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, Normalize

import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import geopandas as gpd

warnings.filterwarnings('ignore')

from utils_qm import (
    VARIAVEIS, ESTACOES_DO_ANO,
    arquivo_corrigido, arquivo_agregado,
    ESTACOES_CSV, SHP_PATH, _PROJECT_DIR,
)

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
DPI         = 200
EXTENT      = [-53.2, -43.9, -25.4, -19.7]
PROJ        = ccrs.PlateCarree()
MARKER_SIZE = 180


# ---------------------------------------------------------------------------
# FUNÇÕES DE MAPA
# ---------------------------------------------------------------------------
def _base_mapa(ax, gdf_sp):
    feat = ShapelyFeature(gdf_sp.geometry, PROJ,
                          facecolor='white', edgecolor='#333333', linewidth=1.0)
    ax.set_extent(EXTENT, crs=PROJ)
    ax.set_facecolor('white')
    ax.add_feature(feat, zorder=2)
    for spine in ax.spines.values():
        spine.set_edgecolor('#AAAAAA'); spine.set_linewidth(0.8)


def _labels(ax, lons, lats, codigos):
    for lon, lat, cod in zip(lons, lats, codigos):
        ax.text(lon + 0.10, lat + 0.10, str(cod),
                fontsize=6.5, fontweight='bold', color='#111111',
                transform=PROJ, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])


def figura_bias_anual(gdf_sp, metricas, cfg, dir_saida):
    """Mapa 1×2: BIAS original e BIAS QM."""
    un        = cfg['unidade']
    lons      = metricas['lon'].values
    lats      = metricas['lat'].values
    codigos   = metricas['estacao'].values
    bias_vals = np.concatenate([metricas['BIAS_orig'].values,
                                 metricas['BIAS_qm'].values])
    absmax    = np.ceil(np.nanpercentile(np.abs(bias_vals), 98) * 10) / 10
    norm_bias = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8.5), subplot_kw={'projection': PROJ},
        facecolor='white',
        gridspec_kw={'wspace': 0.04, 'left': 0.01, 'right': 0.88,
                     'top': 0.88, 'bottom': 0.10},
    )
    sc = None
    for ax, vals, subtit in [
        (axes[0], metricas['BIAS_orig'].values, 'Original'),
        (axes[1], metricas['BIAS_qm'].values,   'Correção por QM'),
    ]:
        _base_mapa(ax, gdf_sp)
        sc = ax.scatter(lons, lats, c=vals, cmap='RdBu_r', norm=norm_bias,
                        s=MARKER_SIZE, zorder=5,
                        edgecolors='#222222', linewidths=0.5, transform=PROJ)
        _labels(ax, lons, lats, codigos)
        ax.set_title(subtit, fontsize=11, fontweight='bold', pad=6)

    cbar_ax = fig.add_axes([0.905, 0.12, 0.009, 0.72])
    cb = fig.colorbar(sc, cax=cbar_ax, extend='both')
    cb.set_label(f'BIAS ({un})', fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f'BIAS Anual — {cfg["label"]} — Estado de São Paulo\n'
        'LOO-CV + QM Mensal Estratificado  (2021–2025)',
        fontsize=13, fontweight='bold', y=0.95,
    )
    arq = os.path.join(dir_saida, 'mapas_bias_anual.png')
    fig.savefig(arq, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Salvo: {arq}")


def figura_cv_sazonal(gdf_sp, df_cv, cfg, dir_saida):
    """Mapa 2×2: CV sazonal do modelo corrigido."""
    col_qm   = cfg['col_qm']
    cv_global, cv_dados = [], {}

    for sigla, meses in ESTACOES_DO_ANO.items():
        cv_est = (
            df_cv[df_cv['mes'].isin(meses)]
            .groupby(['estacao', 'lat', 'lon'])[col_qm]
            .agg(lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan)
            .reset_index().rename(columns={col_qm: 'cv'}).dropna(subset=['cv'])
        )
        cv_dados[sigla] = cv_est
        cv_global.extend(cv_est['cv'].values.tolist())

    vmax_cv = np.ceil(np.nanpercentile(cv_global, 98) * 100) / 100
    norm_cv = Normalize(vmin=0, vmax=vmax_cv)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 12), subplot_kw={'projection': PROJ},
        facecolor='white',
        gridspec_kw={'wspace': 0.04, 'hspace': 0.10,
                     'left': 0.01, 'right': 0.88,
                     'top': 0.90, 'bottom': 0.04},
    )
    sc2 = None
    for ax, sigla in zip(axes.flatten(), list(ESTACOES_DO_ANO.keys())):
        cv_est = cv_dados[sigla]
        _base_mapa(ax, gdf_sp)
        sc2 = ax.scatter(cv_est['lon'].values, cv_est['lat'].values,
                         c=cv_est['cv'].values, cmap='YlOrRd', norm=norm_cv,
                         s=MARKER_SIZE, zorder=5,
                         edgecolors='#222222', linewidths=0.5, transform=PROJ)
        _labels(ax, cv_est['lon'].values, cv_est['lat'].values,
                cv_est['estacao'].values)
        ax.set_title(sigla, fontsize=12, fontweight='bold', pad=6)

    cbar_ax = fig.add_axes([0.90, 0.06, 0.018, 0.80])
    cb = fig.colorbar(sc2, cax=cbar_ax, extend='max')
    cb.set_label('CV = σ / μ  (adimensional)', fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f'Coeficiente de Variação Sazonal — {cfg["label"]} — Estado de São Paulo\n'
        'ICON + Quantile Mapping  (2021–2025)',
        fontsize=13, fontweight='bold', y=0.94,
    )
    arq = os.path.join(dir_saida, 'mapas_cv_sazonal.png')
    fig.savefig(arq, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Salvo: {arq}")


# ---------------------------------------------------------------------------
# EXECUÇÃO
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Script 05 — Mapas Espaciais (Vento + Rajada)")
print("=" * 60)

for arq in [ESTACOES_CSV, SHP_PATH]:
    if not os.path.exists(arq):
        print(f"\n[ERRO] Arquivo não encontrado: {arq}")
        raise SystemExit(1)

est = pd.read_csv(ESTACOES_CSV, sep=';')
est.columns   = est.columns.str.strip()
est['codigo'] = est['codigo'].str.strip()

# Shapefile lido uma única vez
gdf_sp = gpd.read_file(SHP_PATH)

for var, cfg in VARIAVEIS.items():
    arq_agg = arquivo_agregado(var)
    arq_cv  = arquivo_corrigido(var)

    if not os.path.exists(arq_agg) or not os.path.exists(arq_cv):
        print(f"\n  [AVISO] Arquivos de '{var}' não encontrados. "
              f"Execute 01_metricas.py primeiro. Pulando.\n")
        continue

    dir_saida = os.path.join(_PROJECT_DIR, 'graficos', 'mapas_espaciais', var)
    os.makedirs(dir_saida, exist_ok=True)

    print(f"\n  Variável: {cfg['titulo']}  →  {dir_saida}")

    metricas = pd.read_csv(arq_agg, sep=';', decimal=',')
    metricas['estacao'] = metricas['estacao'].astype(str).str.strip()
    metricas = metricas.merge(
        est[['codigo', 'lat', 'lon', 'nome']],
        left_on='estacao', right_on='codigo', how='inner',
    )
    print(f"  Estações com coordenadas: {len(metricas)}")

    df_cv = pd.read_csv(arq_cv, compression='gzip')
    df_cv['estacao'] = df_cv['estacao'].astype(str).str.strip()
    df_cv = df_cv.merge(est[['codigo', 'lat', 'lon']],
                        left_on='estacao', right_on='codigo', how='inner')

    print("  [1/2] Gerando figura de BIAS anual...")
    figura_bias_anual(gdf_sp, metricas, cfg, dir_saida)

    print("  [2/2] Gerando figura de CV sazonal...")
    figura_cv_sazonal(gdf_sp, df_cv, cfg, dir_saida)

print(f"\n{'='*60}\n  CONCLUÍDO\n{'='*60}")
