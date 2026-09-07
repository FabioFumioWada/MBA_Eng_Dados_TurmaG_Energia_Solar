# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Cria tabelas
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- TABELAS: INDICE ONI E PREVISAO ENSO (EL NINO / LA NINA)
# MAGIC -- ORIGEM: NOAA CPC (Climate Prediction Center, EUA)
# MAGIC -- FORMATO: Delta Lake
# MAGIC -- FREQUENCIA DE ATUALIZACAO: mensal
# MAGIC -- OBSERVACAO: camada raw mantem o mesmo conteudo do stage, apenas
# MAGIC -- convertido para tabela (sem traducao ou enriquecimento — isso fica
# MAGIC -- para a camada trusted, no notebook 03_processamento_trusted/09_f_oni_enso).
# MAGIC -- ============================================================
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS mba.raw.oni_noaa
# MAGIC (
# MAGIC     MesRef INT
# MAGIC         COMMENT 'Ano e mes de referencia (AAAAMM), mes central do trimestre movel.',
# MAGIC     Trimestre STRING
# MAGIC         COMMENT 'Codigo do trimestre movel (ex.: JJA = Jun-Jul-Ago), convencao oficial NOAA.',
# MAGIC     OniAnomC DOUBLE
# MAGIC         COMMENT 'Anomalia de temperatura da superficie do mar na regiao Nino 3.4, em graus Celsius.'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Indice ONI (Oceanic Nino Index) historico mensal, 1950-presente. Fonte: NOAA CPC.';
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS mba.raw.previsao_enso_noaa
# MAGIC (
# MAGIC     DataEmissao STRING
# MAGIC         COMMENT 'Mes/ano de emissao do boletim de previsao, como recebido da fonte (em ingles).',
# MAGIC     Trimestre STRING
# MAGIC         COMMENT 'Codigo do trimestre movel + meses por extenso em ingles, como recebido da fonte.',
# MAGIC     ProbLaNinaPct INT
# MAGIC         COMMENT 'Probabilidade de La Nina no trimestre, em percentual.',
# MAGIC     ProbNeutroPct INT
# MAGIC         COMMENT 'Probabilidade de condicao neutra no trimestre, em percentual.',
# MAGIC     ProbElNinoPct INT
# MAGIC         COMMENT 'Probabilidade de El Nino no trimestre, em percentual.'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Previsao probabilistica ENSO (El Nino/La Nina/Neutro) da NOAA CPC, atualizada mensalmente. Fonte: NOAA CPC.';

# COMMAND ----------

# DBTITLE 1,Carregar indice ONI historico
from pyspark.sql import functions as F

df_oni = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "false")
    .load("/Volumes/mba/stage/dados_bruto/oni_noaa/oni_historico_mensal.csv")
    .withColumn("MesRef", F.col("MesRef").cast("int"))
    .withColumn("OniAnomC", F.col("OniAnomC").cast("double"))
    .select("MesRef", "Trimestre", "OniAnomC")
)

(
    df_oni.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("mba.raw.oni_noaa")
)

print(f"mba.raw.oni_noaa: {spark.table('mba.raw.oni_noaa').count():,} linhas")

# COMMAND ----------

# DBTITLE 1,Carregar previsao ENSO atual
df_prev = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "false")
    .load("/Volumes/mba/stage/dados_bruto/oni_noaa/previsao_enso_atual.csv")
    .withColumn("ProbLaNinaPct", F.col("ProbLaNinaPct").cast("int"))
    .withColumn("ProbNeutroPct", F.col("ProbNeutroPct").cast("int"))
    .withColumn("ProbElNinoPct", F.col("ProbElNinoPct").cast("int"))
    .select("DataEmissao", "Trimestre", "ProbLaNinaPct", "ProbNeutroPct", "ProbElNinoPct")
)

(
    df_prev.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("mba.raw.previsao_enso_noaa")
)

print(f"mba.raw.previsao_enso_noaa: {spark.table('mba.raw.previsao_enso_noaa').count():,} linhas")

# COMMAND ----------

dbutils.notebook.exit("OK")