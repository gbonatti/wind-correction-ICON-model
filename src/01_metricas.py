"""
01_metricas_v2.py
=================
Calcula métricas de desempenho do ICON Original vs ICON + QM.
Depende de: utils_qm.py (mesmo diretório)
"""

import pandas as pd
import os
from utils_qm import (
    ANOS, ARQUIVO_CORRIGIDO, ARQUIVO_DETALHADO, ARQUIVO_AGREGADO,
    listar_arquivos_por_estacao, ler_arquivo,
    aplicar_cv_qm, calcular_linha_metricas,
)

# ---------------------------------------------------------------------------
# LEITURA
# ---------------------------------------------------------------------------
arquivos_por_estacao = listar_arquivos_por_estacao()
total = len(arquivos_por_estacao)

print(f"{'='*60}")
print(f"  Métricas v2 — CV Leave-One-Year-Out + QM Mensal")
print(f"{'='*60}")
print(f"  Estações : {total}  |  Anos : {ANOS}\n")

blocos_cv = []

for idx, (eid, arquivos) in enumerate(arquivos_por_estacao.items(), 1):
    print(f"  [{idx/total*100:5.1f}%] {eid} ({idx}/{total})", end='\r')

    frames = []
    for arq in arquivos:
        df, _ = ler_arquivo(arq)
        if df is not None:
            frames.append(df)

    if not frames:
        continue

    df_est = pd.concat(frames, ignore_index=True)
    resultado = aplicar_cv_qm(df_est)
    if resultado is not None:
        blocos_cv.append(resultado)

# ---------------------------------------------------------------------------
# CONSOLIDAÇÃO
# ---------------------------------------------------------------------------
print(f"\n\n  Consolidando {len(blocos_cv)} blocos...")
full_df = pd.concat(blocos_cv, ignore_index=True)

colunas = ['estacao', 'ano', 'mes', 'ciclo', 'hora_int',
           'vento_obs', 'vento_icon', 'vento_icon_qm']
full_df[colunas].to_csv(ARQUIVO_CORRIGIDO, index=False, compression='gzip')
print(f"  Intermediário salvo: {ARQUIVO_CORRIGIDO}")

# ---------------------------------------------------------------------------
# TABELA DETALHADA — estação × mês × ciclo
# ---------------------------------------------------------------------------
print("  Calculando tabela detalhada...")
det = (full_df
       .groupby(['estacao', 'mes', 'ciclo'])
       .apply(calcular_linha_metricas, include_groups=False)
       .reset_index())
det['amostra'] = det['amostra'].astype(int)
det.to_csv(ARQUIVO_DETALHADO, sep=';', index=False, decimal=',')
print(f"  Tabela detalhada  → {ARQUIVO_DETALHADO}")

# ---------------------------------------------------------------------------
# TABELA AGREGADA — uma linha por estação
# ---------------------------------------------------------------------------
print("  Calculando tabela agregada...")
agg = (full_df
       .groupby('estacao')
       .apply(calcular_linha_metricas, include_groups=False)
       .reset_index())
agg['amostra'] = agg['amostra'].astype(int)
agg = agg.sort_values('RMSE_orig', ascending=False)
agg.to_csv(ARQUIVO_AGREGADO, sep=';', index=False, decimal=',')
print(f"  Tabela agregada   → {ARQUIVO_AGREGADO}")

print(f"\n{'='*60}")
print(f"  CONCLUÍDO — {len(full_df):,} registros processados")
print(f"{'='*60}")
