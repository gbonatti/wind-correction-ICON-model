"""
05_mapas_espaciais.py
=====================
Gera 6 mapas de distribuição espacial para o estado de SP usando Cartopy:

  Mapa 1 — BIAS anual: ICON Original
  Mapa 2 — BIAS anual: ICON + QM
  Mapa 3 — CV do vento corrigido — DJF
  Mapa 4 — CV do vento corrigido — MAM
  Mapa 5 — CV do vento corrigido — JJA
  Mapa 6 — CV do vento corrigido — SON

  CV = Coeficiente de Variação = std(vento_icon_qm) / mean(vento_icon_qm)

Estilo: fundo branco, fronteiras pretas, círculos coloridos por valor,
        colorbar horizontal na base — sem interpolação de campo.

Depende de:
  utils_qm.py
  metricas_agregadas_v2.csv   (gerado por 01_metricas_v2.py)
  dados_corrigidos_cv.csv.gz  (gerado por 01_metricas_v2.py)
  estacoes.csv                (colunas: nome;codigo;lat;lon)

Bibliotecas: cartopy, scipy  →  pip install cartopy scipy
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, Normalize

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd

warnings.filterwarnings('ignore')

from utils_qm import (
    ARQUIVO_CORRIGIDO, ARQUIVO_AGREGADO, ESTACOES_DO_ANO,
    ESTACOES_CSV, SHP_PATH, _PROJECT_DIR,
)

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
DIR_SAIDA   = os.path.join(_PROJECT_DIR, 'graficos', 'mapas_espaciais')
DPI         = 200
EXTENT      = [-53.2, -43.9, -25.4, -19.7]   # lon_min, lon_max, lat_min, lat_max
PROJ        = ccrs.PlateCarree()
MARKER_SIZE = 180    # tamanho dos círculos das estações

os.makedirs(DIR_SAIDA, exist_ok=True)


# ---------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE MAPA
# ---------------------------------------------------------------------------
def plotar_mapa(valores, lons, lats, codigos, titulo, subtitulo,
                nome_arquivo, cmap, label_cb,
                vcenter=None, vmin=None, vmax=None):
    """
    Mapa de pontos coloridos sobre SP.
    Estilo: fundo branco, fronteiras cinza-escuro, colorbar horizontal embaixo.
    """
    vals = np.asarray(valores, dtype=float)

    # Escala de cores
    v0 = vmin if vmin is not None else np.nanpercentile(vals, 2)
    v1 = vmax if vmax is not None else np.nanpercentile(vals, 98)
    if vcenter is not None:
        absmax = max(abs(v0), abs(v1))
        norm   = TwoSlopeNorm(vmin=-absmax, vcenter=vcenter, vmax=absmax)
    else:
        norm = Normalize(vmin=v0, vmax=v1)

    # ── Figura: mapa em cima, colorbar em baixo ──────────────────────────
    fig = plt.figure(figsize=(10, 9.5), facecolor='white')

    # Eixo do mapa ocupa ~85% da altura
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.78], projection=PROJ)
    ax.set_extent(EXTENT, crs=PROJ)

    # Fundo branco puro
    ax.set_facecolor('white')

    # Fronteiras — shapefile local SP.shp
    gdf_sp = gpd.read_file(SHP_PATH)
    feat_sp = ShapelyFeature(
        gdf_sp.geometry,
        PROJ,
        facecolor='white',
        edgecolor='#333333',
        linewidth=1.0,
    )
    ax.add_feature(feat_sp, zorder=2)

    # Borda do frame
    for spine in ax.spines.values():
        spine.set_edgecolor('#AAAAAA')
        spine.set_linewidth(0.8)

    # Círculos das estações
    sc = ax.scatter(
        lons, lats,
        c=vals, cmap=cmap, norm=norm,
        s=MARKER_SIZE, zorder=5,
        edgecolors='#222222', linewidths=0.5,
        transform=PROJ,
    )

    # Rótulos das estações
    for lon, lat, cod in zip(lons, lats, codigos):
        ax.text(
            lon + 0.10, lat + 0.10, str(cod),
            fontsize=6.5, fontweight='bold', color='#111111',
            transform=PROJ, zorder=6,
            path_effects=[pe.withStroke(linewidth=2.2, foreground='white')],
        )

    # Título dentro do eixo do mapa
    ax.set_title(f'{titulo}\n{subtitulo}',
                 fontsize=13, fontweight='bold', pad=10,
                 loc='center', color='#111111')

    # ── Colorbar horizontal na base ───────────────────────────────────────
    cbar_ax = fig.add_axes([0.08, 0.07, 0.84, 0.022])
    extend  = 'both' if vcenter is not None else 'max'
    cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', extend=extend)
    cb.set_label(label_cb, fontsize=9, labelpad=4)
    cb.ax.tick_params(labelsize=8)

    fig.savefig(nome_arquivo, dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  Salvo: {nome_arquivo}")


# ---------------------------------------------------------------------------
# LEITURA DOS DADOS
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Script 05 — Mapas Espaciais SP  (Cartopy)")
print("=" * 60)

for arq in [ARQUIVO_AGREGADO, ARQUIVO_CORRIGIDO, ESTACOES_CSV, SHP_PATH]:
    if not os.path.exists(arq):
        print(f"\n[ERRO] Arquivo não encontrado: {arq}")
        raise SystemExit(1)

est = pd.read_csv(ESTACOES_CSV, sep=';')
est.columns  = est.columns.str.strip()
est['codigo'] = est['codigo'].str.strip()

metricas = pd.read_csv(ARQUIVO_AGREGADO, sep=';', decimal=',')
metricas['estacao'] = metricas['estacao'].astype(str).str.strip()
metricas = metricas.merge(
    est[['codigo', 'lat', 'lon', 'nome']],
    left_on='estacao', right_on='codigo', how='inner',
)

print(f"  Carregando {ARQUIVO_CORRIGIDO}...")
df_cv = pd.read_csv(ARQUIVO_CORRIGIDO, compression='gzip')
df_cv['estacao'] = df_cv['estacao'].astype(str).str.strip()
df_cv = df_cv.merge(est[['codigo', 'lat', 'lon']],
                    left_on='estacao', right_on='codigo', how='inner')

print(f"  Estações com coordenadas : {len(metricas)}\n")

# Shapefile lido uma única vez e reutilizado em todas as figuras
gdf_sp  = gpd.read_file(SHP_PATH)
lons    = metricas['lon'].values
lats    = metricas['lat'].values
codigos = metricas['estacao'].values


# ---------------------------------------------------------------------------
# FIGURA 1 — BIAS anual (1 × 2)
# ---------------------------------------------------------------------------
print("  [1/2] Gerando figura de BIAS anual...")

# Escala simétrica compartilhada entre os dois painéis
bias_vals = np.concatenate([metricas['BIAS_orig'].values,
                             metricas['BIAS_qm'].values])
absmax = np.ceil(np.nanpercentile(np.abs(bias_vals), 98) * 10) / 10
norm_bias = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

fig1, axes1 = plt.subplots(
    1, 2, figsize=(18, 8.5),
    subplot_kw={'projection': PROJ},
    facecolor='white',
    gridspec_kw={'wspace': 0.04, 'left': 0.01, 'right': 0.88,
                 'top': 0.88, 'bottom': 0.10},
)

paineis_bias = [
    (axes1[0], metricas['BIAS_orig'].values, 'Original'),
    (axes1[1], metricas['BIAS_qm'].values,   'Correção por QM'),
]

for ax, vals, subtit in paineis_bias:
    feat   = ShapelyFeature(gdf_sp.geometry, PROJ,
                            facecolor='white', edgecolor='#333333', linewidth=1.0)
    ax.set_extent(EXTENT, crs=PROJ)
    ax.set_facecolor('white')
    ax.add_feature(feat, zorder=2)
    for spine in ax.spines.values():
        spine.set_edgecolor('#AAAAAA'); spine.set_linewidth(0.8)

    sc = ax.scatter(lons, lats, c=vals, cmap='RdBu_r', norm=norm_bias,
                    s=MARKER_SIZE, zorder=5,
                    edgecolors='#222222', linewidths=0.5, transform=PROJ)
    for lon, lat, cod in zip(lons, lats, codigos):
        ax.text(lon + 0.10, lat + 0.10, str(cod), fontsize=6.5,
                fontweight='bold', color='#111111', transform=PROJ, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    ax.set_title(subtit, fontsize=11, fontweight='bold', pad=6)

# Colorbar única à direita
cbar_ax1 = fig1.add_axes([0.905, 0.12, 0.009, 0.72])
cb1 = fig1.colorbar(sc, cax=cbar_ax1, extend='both')
cb1.set_label('BIAS (m/s)', fontsize=9, labelpad=6)
cb1.ax.tick_params(labelsize=8)

fig1.suptitle(
    'BIAS Anual do Vento — Estado de São Paulo\nLOO-CV + QM Mensal Estratificado  (2021–2025)',
    fontsize=13, fontweight='bold', y=0.95,
)
arq1 = os.path.join(DIR_SAIDA, 'mapas_bias_anual.png')
fig1.savefig(arq1, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"  Salvo: {arq1}")


# ---------------------------------------------------------------------------
# FIGURA 2 — CV sazonal (2 × 2)
# ---------------------------------------------------------------------------
print("  [2/2] Gerando figura de CV sazonal...")

# Calcula CV de todas as estações do ano para escala comum
cv_global = []
cv_dados  = {}
for sigla, meses in ESTACOES_DO_ANO.items():
    cv_est = (
        df_cv[df_cv['mes'].isin(meses)]
        .groupby(['estacao', 'lat', 'lon'])['vento_icon_qm']
        .agg(lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan)
        .reset_index()
        .rename(columns={'vento_icon_qm': 'cv'})
        .dropna(subset=['cv'])
    )
    cv_dados[sigla] = cv_est
    cv_global.extend(cv_est['cv'].values.tolist())

vmax_cv  = np.ceil(np.nanpercentile(cv_global, 98) * 100) / 100
norm_cv  = Normalize(vmin=0, vmax=vmax_cv)

siglas  = list(ESTACOES_DO_ANO.keys())
fig2, axes2 = plt.subplots(
    2, 2, figsize=(14, 12),
    subplot_kw={'projection': PROJ},
    facecolor='white',
    gridspec_kw={'wspace': 0.04, 'hspace': 0.10,
                 'left': 0.01, 'right': 0.88,
                 'top': 0.90, 'bottom': 0.04},
)

for ax, sigla in zip(axes2.flatten(), siglas):
    cv_est = cv_dados[sigla]
    feat   = ShapelyFeature(gdf_sp.geometry, PROJ,
                            facecolor='white', edgecolor='#333333', linewidth=1.0)
    ax.set_extent(EXTENT, crs=PROJ)
    ax.set_facecolor('white')
    ax.add_feature(feat, zorder=2)
    for spine in ax.spines.values():
        spine.set_edgecolor('#AAAAAA'); spine.set_linewidth(0.8)

    sc2 = ax.scatter(cv_est['lon'].values, cv_est['lat'].values,
                     c=cv_est['cv'].values, cmap='YlOrRd', norm=norm_cv,
                     s=MARKER_SIZE, zorder=5,
                     edgecolors='#222222', linewidths=0.5, transform=PROJ)
    for _, row in cv_est.iterrows():
        ax.text(row['lon'] + 0.10, row['lat'] + 0.10, str(row['estacao']),
                fontsize=6.5, fontweight='bold', color='#111111',
                transform=PROJ, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    ax.set_title(sigla, fontsize=12, fontweight='bold', pad=6)

# Colorbar única à direita
cbar_ax2 = fig2.add_axes([0.90, 0.06, 0.018, 0.80])
cb2 = fig2.colorbar(sc2, cax=cbar_ax2, extend='max')
cb2.set_label('CV = σ / μ  (adimensional)', fontsize=9, labelpad=6)
cb2.ax.tick_params(labelsize=8)

fig2.suptitle(
    'Coeficiente de Variação Sazonal do Vento — Estado de São Paulo\nICON + Quantile Mapping  (2021–2025)',
    fontsize=13, fontweight='bold', y=0.94,
)
arq2 = os.path.join(DIR_SAIDA, 'mapas_cv_sazonal.png')
fig2.savefig(arq2, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"  Salvo: {arq2}")

print(f"\n{'='*60}")
print(f"  CONCLUÍDO — 2 figuras em '{DIR_SAIDA}/'")
print(f"  mapas_bias_anual.png  ← BIAS orig vs QM (1×2)")
print(f"  mapas_cv_sazonal.png  ← CV DJF/MAM/JJA/SON (2×2)")
print(f"{'='*60}")
