# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.refined.f_chuvas(
# MAGIC     id_estacao BIGINT COMMENT 'Chave substituta do fato clima diário',
# MAGIC     anomes int COMMENT 'Ano mes referencia do periodo de chuvas',
# MAGIC     chuva_mm DOUBLE COMMENT 'Precipitação total do dia, em mm'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Fato mensal de chuva por estação meteorológica, agregado a partir das leituras horárias. Granularidade: estação x mes. Fonte: INMET - BDMEP, via mba.raw.clima_inmet.'

# COMMAND ----------

# DBTITLE 1,Agrega a camada raw para o nível diário
from pyspark.sql import functions as F

spark.sql(f"""
            select CAST(date_format(a.data, 'yyyyMM')AS INT) AS anomes
                , b.id_estacao
                , coalesce(sum(precipitacao_total_dia_mm),0) chuva_mm
            from mba.trusted.f_clima_diario a
            join mba.trusted.d_estacao_meterologica b on a.codigo_wmo =b.codigo_wmo
            group by all
          """).createOrReplaceTempView("stg_clima_diario")


# COMMAND ----------

# DBTITLE 1,Grava na tabela trusted (idempotente)
# MAGIC %sql
# MAGIC MERGE INTO mba.refined.f_chuvas AS tgt
# MAGIC USING stg_clima_diario AS src
# MAGIC ON tgt.id_estacao = src.id_estacao and tgt.anomes = src.anomes
# MAGIC
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         tgt.chuva_mm = src.chuva_mm
# MAGIC
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT ( id_estacao, anomes, chuva_mm )
# MAGIC     VALUES ( src.id_estacao, src.anomes, src.chuva_mm );

# COMMAND ----------

dbutils.notebook.exit("Executed")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from mba.refined.f_chuvas a
# MAGIC join mba.trusted.d_estacao_meterologica b on a.id_estacao=b.id_estacao

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nota sobre casamento usina x estacao
# MAGIC A tentativa antiga de ligar cada usina a estacao meteorologica mais
# MAGIC proxima (que estava aqui com bug: os alias t1/t2 nao existiam e as
# MAGIC tabelas nao tem coluna geom) foi corrigida e virou um notebook
# MAGIC dedicado na camada trusted: `03_processamento_trusted/04_d_usina_estacao`.
# MAGIC Ele usa a formula de Haversine com latitude/longitude e grava o
# MAGIC resultado em `mba.trusted.d_usina_estacao`, que e a tabela usada pelo
# MAGIC `05_f_modelo_bandeira_clima`. Nao e necessario rodar nada aqui, ja
# MAGIC esta resolvido la.