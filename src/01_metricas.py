"""
01_metricas.py
==============
Calcula métricas de desempenho do ICON Original vs ICON + QM
para vento médio e rajada de vento.

Saídas por variável (vento / rajada):
  dados/dados_corrigidos_cv_<var>.csv.gz
  dados/metricas_detalhadas_<var>.csv
  dados/metricas_agregadas_<var>.csv

Depende de: utils_qm.py (mesmo diretório)
"""

import pandas as pd
from utils_qm import (
    ANOS, VARIAVEIS,
    arquivo_corrigido, arquivo_detalhado, arquivo_agregado,
    listar_arquivos_por_estacao, ler_arquivo,
    aplicar_cv_qm, calcular_linha_metricas,
)

# ---------------------------------------------------------------------------
# LEITURA — carrega todos os CSVs uma única vez
# ---------------------------------------------------------------------------
arquivos_por_estacao = listar_arquivos_por_estacao()
total = len(arquivos_por_estacao)

print(f"{'='*60}")
print(f"  Métricas — CV Leave-One-Year-Out + QM Mensal")
print(f"{'='*60}")
print(f"  Estações : {total}  |  Anos : {ANOS}\n")

print("  Lendo arquivos...")
frames_por_estacao: dict[str, pd.DataFrame] = {}
for idx, (eid, arquivos) in enumerate(arquivos_por_estacao.items(), 1):
    print(f"  [{idx/total*100:5.1f}%] {eid} ({idx}/{total})", end='\r')
    frames = []
    for arq in arquivos:
        df, _ = ler_arquivo(arq)
        if df is not None:
            frames.append(df)
    if frames:
        frames_por_estacao[eid] = pd.concat(frames, ignore_index=True)

print(f"\n  {len(frames_por_estacao)} estações carregadas.\n")

# ---------------------------------------------------------------------------
# PROCESSAMENTO — uma iteração por variável
# ---------------------------------------------------------------------------
for var, cfg in VARIAVEIS.items():
    col_obs   = cfg['col_obs']
    col_model = cfg['col_model']
    col_qm    = cfg['col_qm']
    label     = cfg['titulo']

    print(f"{'─'*60}")
    print(f"  Variável: {label}")
    print(f"{'─'*60}")

    blocos_cv = []
    n_sem_dados = 0
    for idx, (eid, df_est) in enumerate(frames_por_estacao.items(), 1):
        print(f"  [{idx/total*100:5.1f}%] {eid} ({idx}/{total})", end='\r')

        if col_obs not in df_est.columns or col_model not in df_est.columns:
            n_sem_dados += 1
            continue

        resultado = aplicar_cv_qm(df_est, var=var)
        if resultado is not None:
            blocos_cv.append(resultado)

    if not blocos_cv:
        print(f"\n  [AVISO] Nenhum dado de '{var}' encontrado nos CSVs. "
              f"Verifique as colunas '{col_obs}' e '{col_model}'.\n")
        continue

    if n_sem_dados:
        print(f"\n  [INFO] {n_sem_dados} estação(ões) sem colunas de {var}, ignoradas.")

    # Consolidação
    print(f"\n\n  Consolidando {len(blocos_cv)} blocos ({var})...")
    full_df = pd.concat(blocos_cv, ignore_index=True)

    colunas_base = ['estacao', 'ano', 'mes', 'ciclo', 'hora_int']
    colunas_var  = [col_obs, col_model, col_qm]
    full_df[colunas_base + colunas_var].to_csv(
        arquivo_corrigido(var), index=False, compression='gzip'
    )
    print(f"  Intermediário salvo: {arquivo_corrigido(var)}")

    # Tabela detalhada — estação × mês × ciclo
    print("  Calculando tabela detalhada...")
    fn_metricas = lambda g: calcular_linha_metricas(g, var=var)
    det = (full_df
           .groupby(['estacao', 'mes', 'ciclo'])
           .apply(fn_metricas, include_groups=False)
           .reset_index())
    det['amostra'] = det['amostra'].astype(int)
    det.to_csv(arquivo_detalhado(var), sep=';', index=False, decimal=',')
    print(f"  Tabela detalhada  → {arquivo_detalhado(var)}")

    # Tabela agregada — uma linha por estação
    print("  Calculando tabela agregada...")
    agg = (full_df
           .groupby('estacao')
           .apply(fn_metricas, include_groups=False)
           .reset_index())
    agg['amostra'] = agg['amostra'].astype(int)
    agg = agg.sort_values('RMSE_orig', ascending=False)
    agg.to_csv(arquivo_agregado(var), sep=';', index=False, decimal=',')
    print(f"  Tabela agregada   → {arquivo_agregado(var)}")

    print(f"\n  CONCLUÍDO [{var}] — {len(full_df):,} registros processados\n")

print(f"{'='*60}")
print(f"  PIPELINE CONCLUÍDO")
print(f"{'='*60}")
