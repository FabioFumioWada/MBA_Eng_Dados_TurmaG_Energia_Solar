# Databricks notebook source
# MAGIC %md
# MAGIC ## Dimensao Usina x Estacao Meteorologica
# MAGIC
# MAGIC ### Objetivo
# MAGIC Associar cada usina hidreletrica (mba.trusted.d_usinas) as estacoes
# MAGIC meteorologicas do INMET (mba.trusted.d_estacao_meterologica) mais
# MAGIC proximas geograficamente, para depois agregar o clima dessas estacoes
# MAGIC e usar como preditor no modelo de bandeira tarifaria.
# MAGIC
# MAGIC ### Metodo
# MAGIC Calculamos a distancia entre cada usina e cada estacao usando a
# MAGIC formula de Haversine (latitude/longitude). Regra de casamento:
# MAGIC 1. Mantemos todos os pares usina-estacao com distancia ate 100 km.
# MAGIC 2. Para usinas sem nenhuma estacao dentro de 100 km, usamos a
# MAGIC    estacao mais proxima disponivel como reserva (fallback), para
# MAGIC    garantir que toda usina tenha ao menos uma estacao associada.

# COMMAND ----------

# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.trusted.d_usina_estacao (
# MAGIC     id_usina BIGINT COMMENT 'Chave da usina, referencia a mba.trusted.d_usinas',
# MAGIC     CodCEG STRING COMMENT 'Codigo do empreendimento de geracao',
# MAGIC     cod_estacao STRING COMMENT 'Codigo WMO da estacao meteorologica associada (mba.trusted.d_estacao_meterologica.codigo_wmo)',
# MAGIC     distancia_km DOUBLE COMMENT 'Distancia aproximada entre usina e estacao, em km (formula de Haversine)',
# MAGIC     DatCarga TIMESTAMP COMMENT 'Data e hora do processamento na camada trusted'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Associacao entre usinas hidreletricas e estacoes meteorologicas proximas (raio de 100 km, com fallback para a mais proxima), usada para agregar clima por usina no modelo bandeira x clima.'

# COMMAND ----------

# DBTITLE 1,Calcula distancias e monta pares usina-estacao
spark.sql("""
    WITH distancias AS (
        SELECT
            u.id_usina,
            u.CodCEG,
            e.codigo_wmo AS cod_estacao,
            (6371 * acos(
                LEAST(1.0, GREATEST(-1.0,
                    cos(radians(u.Latitude)) * cos(radians(e.latitude)) *
                    cos(radians(e.longitude) - radians(u.Longitude)) +
                    sin(radians(u.Latitude)) * sin(radians(e.latitude))
                ))
            )) AS distancia_km
        FROM mba.trusted.d_usinas u
        CROSS JOIN mba.trusted.d_estacao_meterologica e
        WHERE u.Latitude IS NOT NULL AND u.Longitude IS NOT NULL
          AND e.latitude IS NOT NULL AND e.longitude IS NOT NULL
    ),
    dentro_do_raio AS (
        SELECT id_usina, CodCEG, cod_estacao, distancia_km
        FROM distancias
        WHERE distancia_km <= 100
    ),
    mais_proxima AS (
        SELECT id_usina, CodCEG, cod_estacao, distancia_km
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id_usina ORDER BY distancia_km) AS rn
            FROM distancias
        )
        WHERE rn = 1
    ),
    sem_estacao_no_raio AS (
        SELECT m.*
        FROM mais_proxima m
        LEFT ANTI JOIN dentro_do_raio d ON m.id_usina = d.id_usina
    )
    SELECT id_usina, CodCEG, cod_estacao, distancia_km FROM dentro_do_raio
    UNION ALL
    SELECT id_usina, CodCEG, cod_estacao, distancia_km FROM sem_estacao_no_raio
""").createOrReplaceTempView("stg_usina_estacao")

print(f"Pares usina-estacao: {spark.table('stg_usina_estacao').count():,}")

# COMMAND ----------

# DBTITLE 1,Grava na tabela trusted (idempotente)
# MAGIC %sql
# MAGIC MERGE INTO mba.trusted.d_usina_estacao AS tgt
# MAGIC USING stg_usina_estacao AS src
# MAGIC ON tgt.id_usina = src.id_usina AND tgt.cod_estacao = src.cod_estacao
# MAGIC
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         tgt.CodCEG = src.CodCEG,
# MAGIC         tgt.distancia_km = src.distancia_km,
# MAGIC         tgt.DatCarga = current_timestamp()
# MAGIC
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT (id_usina, CodCEG, cod_estacao, distancia_km, DatCarga)
# MAGIC     VALUES (src.id_usina, src.CodCEG, src.cod_estacao, src.distancia_km, current_timestamp());

# COMMAND ----------

dbutils.notebook.exit("Executed")
