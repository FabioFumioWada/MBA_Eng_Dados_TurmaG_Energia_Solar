# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.trusted.f_carga_energia_subsistema(
# MAGIC     id_carga_energia_subsistema BIGINT GENERATED ALWAYS AS IDENTITY COMMENT 'Chave substituta do fato Carga de Energia diária',
# MAGIC     id_subsistema STRING COMMENT 'Código do subsistema (N, NE, S, SE)',
# MAGIC     nom_subsistema STRING COMMENT 'Nome do subsistema',
# MAGIC     din_instante DATE COMMENT 'Data da medição',
# MAGIC     val_cargaenergiamwmed DOUBLE COMMENT 'Carga de energia verificada, em MWmed',
# MAGIC     DatCarga TIMESTAMP COMMENT 'Data e hora do processamento na camada trusted'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Fato diário de Carga de Energia por subsistema. Granularidade: subsistema x dia. Fonte: ONS, via mba.raw.carga_energia_subsistema.'

# COMMAND ----------

# DBTITLE 1,Alimenta tabela
# MAGIC %sql
# MAGIC MERGE INTO mba.trusted.f_carga_energia_subsistema AS tgt
# MAGIC USING (
# MAGIC     SELECT
# MAGIC         id_subsistema,
# MAGIC         nom_subsistema,
# MAGIC         din_instante,
# MAGIC         val_cargaenergiamwmed,
# MAGIC         current_timestamp() AS DatCarga
# MAGIC     FROM mba.raw.carga_energia_subsistema
# MAGIC     WHERE id_subsistema IS NOT NULL AND din_instante IS NOT NULL
# MAGIC ) AS src
# MAGIC ON tgt.id_subsistema = src.id_subsistema AND tgt.din_instante = src.din_instante
# MAGIC
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         tgt.val_cargaenergiamwmed = src.val_cargaenergiamwmed,
# MAGIC         tgt.DatCarga = src.DatCarga
# MAGIC
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT (id_subsistema, nom_subsistema, din_instante, val_cargaenergiamwmed, DatCarga)
# MAGIC     VALUES (src.id_subsistema, src.nom_subsistema, src.din_instante, src.val_cargaenergiamwmed, src.DatCarga);

# COMMAND ----------

dbutils.notebook.exit("OK")
