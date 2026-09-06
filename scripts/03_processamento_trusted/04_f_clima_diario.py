# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.trusted.f_clima_diario(
# MAGIC     id_clima_diario BIGINT GENERATED ALWAYS AS IDENTITY COMMENT 'Chave substituta do fato clima diário',
# MAGIC     codigo_wmo STRING COMMENT 'Código OMM/WMO da estação meteorológica',
# MAGIC     estacao STRING COMMENT 'Nome da estação meteorológica',
# MAGIC     uf STRING COMMENT 'UF da estação',
# MAGIC     regiao STRING COMMENT 'Região do Brasil da estação',
# MAGIC     latitude DOUBLE COMMENT 'Latitude da estação, em grau decimal',
# MAGIC     longitude DOUBLE COMMENT 'Longitude da estação, em grau decimal',
# MAGIC     altitude_m DOUBLE COMMENT 'Altitude da estação, em metros',
# MAGIC     data DATE COMMENT 'Data da observação',
# MAGIC     precipitacao_total_dia_mm DOUBLE COMMENT 'Precipitação total do dia, em mm',
# MAGIC     horas_com_leitura_precipitacao BIGINT COMMENT 'Quantidade de leituras horárias de precipitação válidas no dia',
# MAGIC     temperatura_media_c DOUBLE COMMENT 'Temperatura média do ar no dia, em °C',
# MAGIC     temperatura_max_c DOUBLE COMMENT 'Temperatura máxima do ar no dia, em °C',
# MAGIC     temperatura_min_c DOUBLE COMMENT 'Temperatura mínima do ar no dia, em °C',
# MAGIC     umidade_relativa_media_pct DOUBLE COMMENT 'Umidade relativa média do dia, em %',
# MAGIC     vento_velocidade_media_ms DOUBLE COMMENT 'Velocidade média do vento no dia, em m/s',
# MAGIC     vento_rajada_max_ms DOUBLE COMMENT 'Rajada máxima de vento no dia, em m/s',
# MAGIC     DatCarga TIMESTAMP COMMENT 'Data e hora do processamento na camada trusted'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Fato diário de clima por estação meteorológica, agregado a partir das leituras horárias. Granularidade: estação x dia. Fonte: INMET - BDMEP, via mba.raw.clima_inmet.'

# COMMAND ----------

# DBTITLE 1,Agrega a camada raw para o nível diário
from pyspark.sql import functions as F

agregado = (
    spark.table("mba.raw.clima_inmet")
    .groupBy("codigo_wmo", "estacao", "uf", "regiao", "latitude", "longitude", "altitude_m", "data")
    .agg(
        F.sum("precipitacao_total_mm").alias("precipitacao_total_dia_mm"),
        F.count("precipitacao_total_mm").alias("horas_com_leitura_precipitacao"),
        F.avg("temperatura_ar_c").alias("temperatura_media_c"),
        F.max("temperatura_ar_c").alias("temperatura_max_c"),
        F.min("temperatura_ar_c").alias("temperatura_min_c"),
        F.avg("umidade_relativa_pct").alias("umidade_relativa_media_pct"),
        F.avg("vento_velocidade_ms").alias("vento_velocidade_media_ms"),
        F.max("vento_rajada_ms").alias("vento_rajada_max_ms"),
    )
    .withColumn("DatCarga", F.current_timestamp())
)

agregado.createOrReplaceTempView("stg_clima_diario")
print(f"Linhas agregadas (estação x dia): {agregado.count():,}")

# COMMAND ----------

# DBTITLE 1,Grava na tabela trusted (idempotente)
# MAGIC %sql
# MAGIC MERGE INTO mba.trusted.f_clima_diario AS tgt
# MAGIC USING stg_clima_diario AS src
# MAGIC ON tgt.codigo_wmo = src.codigo_wmo AND tgt.data = src.data
# MAGIC
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         tgt.estacao = src.estacao,
# MAGIC         tgt.uf = src.uf,
# MAGIC         tgt.regiao = src.regiao,
# MAGIC         tgt.latitude = src.latitude,
# MAGIC         tgt.longitude = src.longitude,
# MAGIC         tgt.altitude_m = src.altitude_m,
# MAGIC         tgt.precipitacao_total_dia_mm = src.precipitacao_total_dia_mm,
# MAGIC         tgt.horas_com_leitura_precipitacao = src.horas_com_leitura_precipitacao,
# MAGIC         tgt.temperatura_media_c = src.temperatura_media_c,
# MAGIC         tgt.temperatura_max_c = src.temperatura_max_c,
# MAGIC         tgt.temperatura_min_c = src.temperatura_min_c,
# MAGIC         tgt.umidade_relativa_media_pct = src.umidade_relativa_media_pct,
# MAGIC         tgt.vento_velocidade_media_ms = src.vento_velocidade_media_ms,
# MAGIC         tgt.vento_rajada_max_ms = src.vento_rajada_max_ms,
# MAGIC         tgt.DatCarga = src.DatCarga
# MAGIC
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT (
# MAGIC         codigo_wmo, estacao, uf, regiao, latitude, longitude, altitude_m, data,
# MAGIC         precipitacao_total_dia_mm, horas_com_leitura_precipitacao,
# MAGIC         temperatura_media_c, temperatura_max_c, temperatura_min_c,
# MAGIC         umidade_relativa_media_pct, vento_velocidade_media_ms, vento_rajada_max_ms, DatCarga
# MAGIC     )
# MAGIC     VALUES (
# MAGIC         src.codigo_wmo, src.estacao, src.uf, src.regiao, src.latitude, src.longitude, src.altitude_m, src.data,
# MAGIC         src.precipitacao_total_dia_mm, src.horas_com_leitura_precipitacao,
# MAGIC         src.temperatura_media_c, src.temperatura_max_c, src.temperatura_min_c,
# MAGIC         src.umidade_relativa_media_pct, src.vento_velocidade_media_ms, src.vento_rajada_max_ms, src.DatCarga
# MAGIC     );

# COMMAND ----------

dbutils.notebook.exit("Executed")

# COMMAND ----------

display(
    spark.sql("""
        SELECT *
        FROM mba.trusted.f_clima_diario
        LIMIT 20
    """)
)

spark.sql("""
    SELECT COUNT(*) AS quantidade_registros, COUNT(DISTINCT codigo_wmo) AS estacoes
    FROM mba.trusted.f_clima_diario
""").show()
