# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## 09 - Indice ONI e Previsao ENSO (Trusted)
# MAGIC
# MAGIC Camada trusted: dado tipado, padronizado, **traduzido para portugues** e com
# MAGIC as colunas de apoio (data, ano/mes, fase do fenomeno) ja calculadas, para
# MAGIC uso direto no modelo preditivo e em qualquer relatorio/apresentacao do
# MAGIC grupo. A camada raw (`mba.raw.oni_noaa` e `mba.raw.previsao_enso_noaa`)
# MAGIC mantem o conteudo original em ingles, exatamente como veio da NOAA — a
# MAGIC traducao e o enriquecimento acontecem so aqui, seguindo o mesmo padrao das
# MAGIC outras fontes do projeto (raw = espelho da fonte, trusted = dado pronto
# MAGIC para uso).

# COMMAND ----------

# DBTITLE 1,Cria e alimenta f_oni_enso (historico)
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE mba.trusted.f_oni_enso
# MAGIC COMMENT 'Indice ONI historico mensal, tipado e com fase do ENSO classificada. Granularidade: 1 linha por mes. Fonte: mba.raw.oni_noaa.'
# MAGIC AS
# MAGIC SELECT
# MAGIC     MesRef,
# MAGIC     CAST(MesRef DIV 100 AS INT) AS Ano,
# MAGIC     CAST(MesRef % 100 AS INT) AS Mes,
# MAGIC     make_date(CAST(MesRef DIV 100 AS INT), CAST(MesRef % 100 AS INT), 1) AS DataRef,
# MAGIC     Trimestre,
# MAGIC     CAST(OniAnomC AS DOUBLE) AS OniAnomC,
# MAGIC     CASE
# MAGIC         WHEN OniAnomC >= 0.5 THEN 'El Niño'
# MAGIC         WHEN OniAnomC <= -0.5 THEN 'La Niña'
# MAGIC         ELSE 'Neutro'
# MAGIC     END AS FaseEnso
# MAGIC FROM mba.raw.oni_noaa
# MAGIC ORDER BY MesRef;

# COMMAND ----------

# DBTITLE 1,Traduz e alimenta f_previsao_enso (previsao atual)
# A traducao do trimestre (codigo + meses por extenso) depende da regra oficial
# NOAA de qual ano cada trimestre movel pertence (o mes central "vira o ano"
# quando passa de dezembro para janeiro) — por isso é feita em Python, e nao em SQL puro.

from pyspark.sql import functions as F, types as T

TRIMESTRE_MESES_EN = {
    "DJF": ["Dec", "Jan", "Feb"], "JFM": ["Jan", "Feb", "Mar"], "FMA": ["Feb", "Mar", "Apr"],
    "MAM": ["Mar", "Apr", "May"], "AMJ": ["Apr", "May", "Jun"], "MJJ": ["May", "Jun", "Jul"],
    "JJA": ["Jun", "Jul", "Aug"], "JAS": ["Jul", "Aug", "Sep"], "ASO": ["Aug", "Sep", "Oct"],
    "SON": ["Sep", "Oct", "Nov"], "OND": ["Oct", "Nov", "Dec"], "NDJ": ["Nov", "Dec", "Jan"],
}
EN_PT_MES = {
    "Jan": "Jan", "Feb": "Fev", "Mar": "Mar", "Apr": "Abr", "May": "Mai", "Jun": "Jun",
    "Jul": "Jul", "Aug": "Ago", "Sep": "Set", "Oct": "Out", "Nov": "Nov", "Dec": "Dez",
}
MES_NUM_EN = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
              "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

df_raw = spark.table("mba.raw.previsao_enso_noaa").toPandas()
codigos = df_raw["Trimestre"].str.split().str[0].tolist()

ano_emissao = int(df_raw["DataEmissao"].iloc[0].split()[-1])
ano_atual = ano_emissao
mes_central_anterior = None
descricoes, mes_ref_central_lista = [], []

for codigo in codigos:
    mes_central = SEASON_TO_MONTH[codigo]
    if mes_central_anterior is not None and mes_central < mes_central_anterior:
        ano_atual += 1
    mes_central_anterior = mes_central

    m0, m1, m2 = [MES_NUM_EN[m] for m in TRIMESTRE_MESES_EN[codigo]]
    y1 = ano_atual
    y0 = y1 - 1 if m0 > m1 else y1
    y2 = y1 + 1 if m2 < m1 else y1
    meses_pt = [EN_PT_MES[m] for m in TRIMESTRE_MESES_EN[codigo]]
    descricao = f"{codigo} ({'-'.join(meses_pt)}) — {meses_pt[0]}/{y0} a {meses_pt[2]}/{y2}"
    descricoes.append(descricao)
    mes_ref_central_lista.append(ano_atual * 100 + mes_central)

df_raw["TrimestreCodigo"] = codigos
df_raw["TrimestreDescricaoPt"] = descricoes
df_raw["MesCentralRef"] = mes_ref_central_lista
df_raw["DataEmissao"] = "Agosto de 2026" if ano_emissao == 2026 else df_raw["DataEmissao"]

df_final = df_raw[[
    "DataEmissao", "TrimestreCodigo", "TrimestreDescricaoPt", "MesCentralRef",
    "ProbLaNinaPct", "ProbNeutroPct", "ProbElNinoPct",
]]

spark_df = spark.createDataFrame(df_final)
spark_df = (
    spark_df
    .withColumn("AnoRef", (F.col("MesCentralRef") / 100).cast("int"))
    .withColumn("MesRef", (F.col("MesCentralRef") % 100).cast("int"))
    .withColumn("ProbLaNinaPct", F.col("ProbLaNinaPct").cast("double"))
    .withColumn("ProbNeutroPct", F.col("ProbNeutroPct").cast("double"))
    .withColumn("ProbElNinoPct", F.col("ProbElNinoPct").cast("double"))
    .orderBy("MesCentralRef")
)

(
    spark_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("mba.trusted.f_previsao_enso")
)

print(f"mba.trusted.f_previsao_enso: {spark.table('mba.trusted.f_previsao_enso').count():,} linhas")

# COMMAND ----------

dbutils.notebook.exit("OK")