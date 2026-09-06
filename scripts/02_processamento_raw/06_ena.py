# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- TABELA: ENA - ENERGIA NATURAL AFLUENTE POR SUBSISTEMA
# MAGIC -- ORIGEM: ONS - Operador Nacional do Sistema Elétrico
# MAGIC -- DATASET: ena_subsistema_di (dados abertos ONS)
# MAGIC -- FORMATO: Delta Lake
# MAGIC -- FREQUÊNCIA DE ATUALIZAÇÃO: Diária
# MAGIC -- ============================================================
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS mba.raw.ena_subsistema
# MAGIC (
# MAGIC     id_subsistema STRING
# MAGIC         COMMENT 'Código do subsistema (N, NE, S, SE).',
# MAGIC     nom_subsistema STRING
# MAGIC         COMMENT 'Nome do subsistema.',
# MAGIC     ena_data DATE
# MAGIC         COMMENT 'Data da medição da energia natural afluente.',
# MAGIC     ena_bruta_regiao_mwmed DOUBLE
# MAGIC         COMMENT 'Energia natural afluente bruta da região, em MWmed.',
# MAGIC     ena_bruta_regiao_percentualmlt DOUBLE
# MAGIC         COMMENT 'Energia natural afluente bruta da região, em percentual da MLT (Média de Longo Termo).',
# MAGIC     ena_armazenavel_regiao_mwmed DOUBLE
# MAGIC         COMMENT 'Energia natural afluente armazenável da região, em MWmed.',
# MAGIC     ena_armazenavel_regiao_percentualmlt DOUBLE
# MAGIC         COMMENT 'Energia natural afluente armazenável da região, em percentual da MLT.',
# MAGIC     NmArquivoCarga STRING
# MAGIC         COMMENT 'Nome do arquivo de origem utilizado na carga.',
# MAGIC     DatCarga TIMESTAMP
# MAGIC         COMMENT 'Data e hora em que o registro foi carregado na tabela Delta.'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Energia Natural Afluente (ENA) diária por subsistema. Fonte: ONS - Dados Abertos. Dataset: ena_subsistema_di.';

# COMMAND ----------

# DBTITLE 1,Carregar
from pyspark.sql import functions as F

# ============================================================
# 1. CAMINHO DOS ARQUIVOS DE ORIGEM (todos os anos, mesma estrutura)
# ============================================================

caminho_arquivos = "/Volumes/mba/stage/dados_bruto/ena/"

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
    .withColumn("ena_data", F.to_date(F.col("ena_data"), "yyyy-MM-dd"))
    .withColumn("ena_bruta_regiao_mwmed", F.col("ena_bruta_regiao_mwmed").cast("double"))
    .withColumn("ena_bruta_regiao_percentualmlt", F.col("ena_bruta_regiao_percentualmlt").cast("double"))
    .withColumn("ena_armazenavel_regiao_mwmed", F.col("ena_armazenavel_regiao_mwmed").cast("double"))
    .withColumn("ena_armazenavel_regiao_percentualmlt", F.col("ena_armazenavel_regiao_percentualmlt").cast("double"))
    .withColumn("DatCarga", F.current_timestamp())
    .select(
        "id_subsistema",
        "nom_subsistema",
        "ena_data",
        "ena_bruta_regiao_mwmed",
        "ena_bruta_regiao_percentualmlt",
        "ena_armazenavel_regiao_mwmed",
        "ena_armazenavel_regiao_percentualmlt",
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
    .saveAsTable("mba.raw.ena_subsistema")
)

print(f"Linhas gravadas: {spark.table('mba.raw.ena_subsistema').count():,}")

# COMMAND ----------

dbutils.notebook.exit("OK")
