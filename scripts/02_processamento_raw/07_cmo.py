# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- TABELA: CMO - CUSTO MARGINAL DE OPERAÇÃO SEMANAL POR SUBSISTEMA
# MAGIC -- ORIGEM: ONS - Operador Nacional do Sistema Elétrico
# MAGIC -- DATASET: cmo_se (dados abertos ONS)
# MAGIC -- FORMATO: Delta Lake
# MAGIC -- FREQUÊNCIA DE ATUALIZAÇÃO: Semanal
# MAGIC -- ============================================================
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS mba.raw.cmo_subsistema
# MAGIC (
# MAGIC     id_subsistema STRING
# MAGIC         COMMENT 'Código do subsistema (N, NE, S, SE).',
# MAGIC     nom_subsistema STRING
# MAGIC         COMMENT 'Nome do subsistema.',
# MAGIC     din_instante DATE
# MAGIC         COMMENT 'Data de início da semana operativa.',
# MAGIC     val_cmomediasemanal DOUBLE
# MAGIC         COMMENT 'Custo Marginal de Operação médio da semana, em R$/MWh.',
# MAGIC     val_cmoleve DOUBLE
# MAGIC         COMMENT 'Custo Marginal de Operação no patamar de carga leve, em R$/MWh.',
# MAGIC     val_cmomedia DOUBLE
# MAGIC         COMMENT 'Custo Marginal de Operação no patamar de carga média, em R$/MWh.',
# MAGIC     val_cmopesada DOUBLE
# MAGIC         COMMENT 'Custo Marginal de Operação no patamar de carga pesada, em R$/MWh.',
# MAGIC     NmArquivoCarga STRING
# MAGIC         COMMENT 'Nome do arquivo de origem utilizado na carga.',
# MAGIC     DatCarga TIMESTAMP
# MAGIC         COMMENT 'Data e hora em que o registro foi carregado na tabela Delta.'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Custo Marginal de Operação (CMO) semanal por subsistema. Fonte: ONS - Dados Abertos. Dataset: cmo_se.';

# COMMAND ----------

# DBTITLE 1,Carregar
from pyspark.sql import functions as F

# ============================================================
# 1. CAMINHO DOS ARQUIVOS DE ORIGEM (todos os anos, mesma estrutura)
# ============================================================

caminho_arquivos = "/Volumes/mba/stage/dados_bruto/cmo/"

# ============================================================
# 2. LEITURA DE TODOS OS CSVs DO DIRETÓRIO
# ============================================================

df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ";")
    .option("encoding", "UTF-8")
    .option("inferSchema", "false")
    .load(caminho_arquivos)
    .withColumn("NmArquivoCarga", F.col("_metadata.file_name"))
)

# ============================================================
# 3. PADRONIZAÇÃO DOS TIPOS
# ============================================================

df_final = (
    df
    .withColumn("din_instante", F.to_date(F.col("din_instante"), "yyyy-MM-dd"))
    .withColumn("val_cmomediasemanal", F.col("val_cmomediasemanal").cast("double"))
    .withColumn("val_cmoleve", F.col("val_cmoleve").cast("double"))
    .withColumn("val_cmomedia", F.col("val_cmomedia").cast("double"))
    .withColumn("val_cmopesada", F.col("val_cmopesada").cast("double"))
    .withColumn("DatCarga", F.current_timestamp())
    .select(
        "id_subsistema",
        "nom_subsistema",
        "din_instante",
        "val_cmomediasemanal",
        "val_cmoleve",
        "val_cmomedia",
        "val_cmopesada",
        "NmArquivoCarga",
        "DatCarga",
    )
)

# ============================================================
# 4. GRAVAÇÃO NA DELTA TABLE
# ============================================================
(
    df_final.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("mba.raw.cmo_subsistema")
)

print(f"Linhas gravadas: {spark.table('mba.raw.cmo_subsistema').count():,}")

# COMMAND ----------

dbutils.notebook.exit("OK")
