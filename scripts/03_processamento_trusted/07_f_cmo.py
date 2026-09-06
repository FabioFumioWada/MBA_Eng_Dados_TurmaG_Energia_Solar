# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.trusted.f_cmo_subsistema(
# MAGIC     id_cmo_subsistema BIGINT GENERATED ALWAYS AS IDENTITY COMMENT 'Chave substituta do fato CMO semanal',
# MAGIC     id_subsistema STRING COMMENT 'Código do subsistema (N, NE, S, SE)',
# MAGIC     nom_subsistema STRING COMMENT 'Nome do subsistema',
# MAGIC     din_instante DATE COMMENT 'Data de início da semana operativa',
# MAGIC     val_cmomediasemanal DOUBLE COMMENT 'CMO médio da semana, em R$/MWh',
# MAGIC     val_cmoleve DOUBLE COMMENT 'CMO no patamar de carga leve, em R$/MWh',
# MAGIC     val_cmomedia DOUBLE COMMENT 'CMO no patamar de carga média, em R$/MWh',
# MAGIC     val_cmopesada DOUBLE COMMENT 'CMO no patamar de carga pesada, em R$/MWh',
# MAGIC     DatCarga TIMESTAMP COMMENT 'Data e hora do processamento na camada trusted'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Fato semanal de Custo Marginal de Operação (CMO) por subsistema. Granularidade: subsistema x semana. Fonte: ONS, via mba.raw.cmo_subsistema.'

# COMMAND ----------

# DBTITLE 1,Alimenta tabela
# MAGIC %sql
# MAGIC MERGE INTO mba.trusted.f_cmo_subsistema AS tgt
# MAGIC USING (
# MAGIC     SELECT
# MAGIC         id_subsistema,
# MAGIC         nom_subsistema,
# MAGIC         din_instante,
# MAGIC         val_cmomediasemanal,
# MAGIC         val_cmoleve,
# MAGIC         val_cmomedia,
# MAGIC         val_cmopesada,
# MAGIC         current_timestamp() AS DatCarga
# MAGIC     FROM mba.raw.cmo_subsistema
# MAGIC     WHERE id_subsistema IS NOT NULL AND din_instante IS NOT NULL
# MAGIC ) AS src
# MAGIC ON tgt.id_subsistema = src.id_subsistema AND tgt.din_instante = src.din_instante
# MAGIC
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         tgt.val_cmomediasemanal = src.val_cmomediasemanal,
# MAGIC         tgt.val_cmoleve = src.val_cmoleve,
# MAGIC         tgt.val_cmomedia = src.val_cmomedia,
# MAGIC         tgt.val_cmopesada = src.val_cmopesada,
# MAGIC         tgt.DatCarga = src.DatCarga
# MAGIC
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT (id_subsistema, nom_subsistema, din_instante, val_cmomediasemanal, val_cmoleve, val_cmomedia, val_cmopesada, DatCarga)
# MAGIC     VALUES (src.id_subsistema, src.nom_subsistema, src.din_instante, src.val_cmomediasemanal, src.val_cmoleve, src.val_cmomedia, src.val_cmopesada, src.DatCarga);

# COMMAND ----------

dbutils.notebook.exit("OK")
