# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- TABELA: EAR - ENERGIA ARMAZENADA NOS RESERVATÓRIOS POR SUBSISTEMA
# MAGIC -- ORIGEM: ONS - Operador Nacional do Sistema Elétrico
# MAGIC -- DATASET: ear_subsistema_di (dados abertos ONS)
# MAGIC -- FORMATO: Delta Lake
# MAGIC -- FREQUÊNCIA DE ATUALIZAÇÃO: Diária
# MAGIC -- ============================================================
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS mba.raw.ear_subsistema
# MAGIC (
# MAGIC     id_subsistema STRING
# MAGIC         COMMENT 'Código do subsistema (N, NE, S, SE).',
# MAGIC     nom_subsistema STRING
# MAGIC         COMMENT 'Nome do subsistema.',
# MAGIC     ear_data DATE
# MAGIC         COMMENT 'Data da medição da energia armazenada.',
# MAGIC     ear_max_subsistema DOUBLE
# MAGIC         COMMENT 'Energia armazenável máxima do subsistema, em MWmês.',
# MAGIC     ear_verif_subsistema_mwmes DOUBLE
# MAGIC         COMMENT 'Energia armazenada verificada no subsistema, em MWmês.',
# MAGIC     ear_verif_subsistema_percentual DOUBLE
# MAGIC         COMMENT 'Energia armazenada verificada no subsistema, em percentual da capacidade máxima.',
# MAGIC     NmArquivoCarga STRING
# MAGIC         COMMENT 'Nome do arquivo de origem utilizado na carga.',
# MAGIC     DatCarga TIMESTAMP
# MAGIC         COMMENT 'Data e hora em que o registro foi carregado na tabela Delta.'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Energia Armazenada nos Reservatórios (EAR) diária por subsistema. Fonte: ONS - Dados Abertos. Dataset: ear_subsistema_di.';

# COMMAND ----------

# DBTITLE 1,Carregar
from pyspark.sql import functions as F

# ============================================================
# 1. CAMINHO DOS ARQUIVOS DE ORIGEM (todos os anos, mesma estrutura)
# ============================================================

caminho_arquivos = "/Volumes/mba/stage/dados_bruto/ear/"

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
    .withColumn("ear_data", F.to_date(F.col("ear_data"), "yyyy-MM-dd"))
    .withColumn("ear_max_subsistema", F.col("ear_max_subsistema").cast("double"))
    .withColumn("ear_verif_subsistema_mwmes", F.col("ear_verif_subsistema_mwmes").cast("double"))
    .withColumn("ear_verif_subsistema_percentual", F.col("ear_verif_subsistema_percentual").cast("double"))
    .withColumn("DatCarga", F.current_timestamp())
    .select(
        "id_subsistema",
        "nom_subsistema",
        "ear_data",
        "ear_max_subsistema",
        "ear_verif_subsistema_mwmes",
        "ear_verif_subsistema_percentual",
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
    .saveAsTable("mba.raw.ear_subsistema")
)

print(f"Linhas gravadas: {spark.table('mba.raw.ear_subsistema').count():,}")

# COMMAND ----------

dbutils.notebook.exit("OK")
