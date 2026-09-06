# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.trusted.f_ear_subsistema(
# MAGIC     id_ear_subsistema BIGINT GENERATED ALWAYS AS IDENTITY COMMENT 'Chave substituta do fato EAR diário',
# MAGIC     id_subsistema STRING COMMENT 'Código do subsistema (N, NE, S, SE)',
# MAGIC     nom_subsistema STRING COMMENT 'Nome do subsistema',
# MAGIC     ear_data DATE COMMENT 'Data da medição',
# MAGIC     ear_max_subsistema DOUBLE COMMENT 'Energia armazenável máxima do subsistema, em MWmês',
# MAGIC     ear_verif_subsistema_mwmes DOUBLE COMMENT 'Energia armazenada verificada, em MWmês',
# MAGIC     ear_verif_subsistema_percentual DOUBLE COMMENT 'Energia armazenada verificada, em % da capacidade máxima',
# MAGIC     DatCarga TIMESTAMP COMMENT 'Data e hora do processamento na camada trusted'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Fato diário de Energia Armazenada nos Reservatórios (EAR) por subsistema. Granularidade: subsistema x dia. Fonte: ONS, via mba.raw.ear_subsistema.'

# COMMAND ----------

# DBTITLE 1,Alimenta tabela
# MAGIC %sql
# MAGIC MERGE INTO mba.trusted.f_ear_subsistema AS tgt
# MAGIC USING (
# MAGIC     SELECT
# MAGIC         id_subsistema,
# MAGIC         nom_subsistema,
# MAGIC         ear_data,
# MAGIC         ear_max_subsistema,
# MAGIC         ear_verif_subsistema_mwmes,
# MAGIC         ear_verif_subsistema_percentual,
# MAGIC         current_timestamp() AS DatCarga
# MAGIC     FROM mba.raw.ear_subsistema
# MAGIC     WHERE id_subsistema IS NOT NULL AND ear_data IS NOT NULL
# MAGIC ) AS src
# MAGIC ON tgt.id_subsistema = src.id_subsistema AND tgt.ear_data = src.ear_data
# MAGIC
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         tgt.ear_max_subsistema = src.ear_max_subsistema,
# MAGIC         tgt.ear_verif_subsistema_mwmes = src.ear_verif_subsistema_mwmes,
# MAGIC         tgt.ear_verif_subsistema_percentual = src.ear_verif_subsistema_percentual,
# MAGIC         tgt.DatCarga = src.DatCarga
# MAGIC
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT (id_subsistema, nom_subsistema, ear_data, ear_max_subsistema, ear_verif_subsistema_mwmes, ear_verif_subsistema_percentual, DatCarga)
# MAGIC     VALUES (src.id_subsistema, src.nom_subsistema, src.ear_data, src.ear_max_subsistema, src.ear_verif_subsistema_mwmes, src.ear_verif_subsistema_percentual, src.DatCarga);

# COMMAND ----------

dbutils.notebook.exit("OK")
