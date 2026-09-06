# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.trusted.f_ena_subsistema(
# MAGIC     id_ena_subsistema BIGINT GENERATED ALWAYS AS IDENTITY COMMENT 'Chave substituta do fato ENA diário',
# MAGIC     id_subsistema STRING COMMENT 'Código do subsistema (N, NE, S, SE)',
# MAGIC     nom_subsistema STRING COMMENT 'Nome do subsistema',
# MAGIC     ena_data DATE COMMENT 'Data da medição',
# MAGIC     ena_bruta_regiao_mwmed DOUBLE COMMENT 'Energia natural afluente bruta, em MWmed',
# MAGIC     ena_bruta_regiao_percentualmlt DOUBLE COMMENT 'Energia natural afluente bruta, em % da MLT',
# MAGIC     ena_armazenavel_regiao_mwmed DOUBLE COMMENT 'Energia natural afluente armazenável, em MWmed',
# MAGIC     ena_armazenavel_regiao_percentualmlt DOUBLE COMMENT 'Energia natural afluente armazenável, em % da MLT',
# MAGIC     DatCarga TIMESTAMP COMMENT 'Data e hora do processamento na camada trusted'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Fato diário de Energia Natural Afluente (ENA) por subsistema. Granularidade: subsistema x dia. Fonte: ONS, via mba.raw.ena_subsistema.'

# COMMAND ----------

# DBTITLE 1,Alimenta tabela
# MAGIC %sql
# MAGIC MERGE INTO mba.trusted.f_ena_subsistema AS tgt
# MAGIC USING (
# MAGIC     SELECT
# MAGIC         id_subsistema,
# MAGIC         nom_subsistema,
# MAGIC         ena_data,
# MAGIC         ena_bruta_regiao_mwmed,
# MAGIC         ena_bruta_regiao_percentualmlt,
# MAGIC         ena_armazenavel_regiao_mwmed,
# MAGIC         ena_armazenavel_regiao_percentualmlt,
# MAGIC         current_timestamp() AS DatCarga
# MAGIC     FROM mba.raw.ena_subsistema
# MAGIC     WHERE id_subsistema IS NOT NULL AND ena_data IS NOT NULL
# MAGIC ) AS src
# MAGIC ON tgt.id_subsistema = src.id_subsistema AND tgt.ena_data = src.ena_data
# MAGIC
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         tgt.ena_bruta_regiao_mwmed = src.ena_bruta_regiao_mwmed,
# MAGIC         tgt.ena_bruta_regiao_percentualmlt = src.ena_bruta_regiao_percentualmlt,
# MAGIC         tgt.ena_armazenavel_regiao_mwmed = src.ena_armazenavel_regiao_mwmed,
# MAGIC         tgt.ena_armazenavel_regiao_percentualmlt = src.ena_armazenavel_regiao_percentualmlt,
# MAGIC         tgt.DatCarga = src.DatCarga
# MAGIC
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT (id_subsistema, nom_subsistema, ena_data, ena_bruta_regiao_mwmed, ena_bruta_regiao_percentualmlt, ena_armazenavel_regiao_mwmed, ena_armazenavel_regiao_percentualmlt, DatCarga)
# MAGIC     VALUES (src.id_subsistema, src.nom_subsistema, src.ena_data, src.ena_bruta_regiao_mwmed, src.ena_bruta_regiao_percentualmlt, src.ena_armazenavel_regiao_mwmed, src.ena_armazenavel_regiao_percentualmlt, src.DatCarga);

# COMMAND ----------

dbutils.notebook.exit("OK")
