# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- TABELA: CARGA DE ENERGIA DIÁRIA POR SUBSISTEMA
# MAGIC -- ORIGEM: ONS - Operador Nacional do Sistema Elétrico
# MAGIC -- DATASET: carga_energia_di (dados abertos ONS)
# MAGIC -- FORMATO: Delta Lake
# MAGIC -- FREQUÊNCIA DE ATUALIZAÇÃO: Diária
# MAGIC -- ============================================================
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS mba.raw.carga_energia_subsistema
# MAGIC (
# MAGIC     id_subsistema STRING
# MAGIC         COMMENT 'Código do subsistema (N, NE, S, SE).',
# MAGIC     nom_subsistema STRING
# MAGIC         COMMENT 'Nome do subsistema.',
# MAGIC     din_instante DATE
# MAGIC         COMMENT 'Data da medição da carga de energia.',
# MAGIC     val_cargaenergiamwmed DOUBLE
# MAGIC         COMMENT 'Carga de energia verificada no subsistema, em MWmed.',
# MAGIC     NmArquivoCarga STRING
# MAGIC         COMMENT 'Nome do arquivo de origem utilizado na carga.',
# MAGIC     DatCarga TIMESTAMP
# MAGIC         COMMENT 'Data e hora em que o registro foi carregado na tabela Delta.'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Carga de energia diária por subsistema. Fonte: ONS - Dados Abertos. Dataset: carga_energia_di.';

# COMMAND ----------

# DBTITLE 1,Carregar
from pyspark.sql import functions as F

# ============================================================
# 1. CAMINHO DOS ARQUIVOS DE ORIGEM (todos os anos, mesma estrutura)
# ============================================================

caminho_arquivos = "/Volumes/mba/stage/dados_bruto/carga/"

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
    .withColumn(
        "din_instante",
        F.coalesce(
            F.expr("try_to_date(din_instante, 'yyyy-MM-dd')"),
            F.expr("try_to_date(din_instante, 'yyyy-MM-dd HH:mm:ss')"),
        )
    )
    .withColumn("val_cargaenergiamwmed", F.col("val_cargaenergiamwmed").cast("double"))
    .withColumn("DatCarga", F.current_timestamp())
    .select(
        "id_subsistema",
        "nom_subsistema",
        "din_instante",
        "val_cargaenergiamwmed",
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
    .saveAsTable("mba.raw.carga_energia_subsistema")
)

print(f"Linhas gravadas: {spark.table('mba.raw.carga_energia_subsistema').count():,}")

# COMMAND ----------

dbutils.notebook.exit("OK")
