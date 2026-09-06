# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.refined.painel_mensal_bandeira_hidrologia(
# MAGIC     MesCompetencia INT COMMENT 'Mes de competencia, formato YYYYMM, chave do painel',
# MAGIC     AnoMes STRING COMMENT 'Mes de competencia, formato YYYY-MM, para leitura',
# MAGIC     NivelBandeira TINYINT COMMENT 'Nivel ordinal da bandeira: 0=Verde, 1=Amarela, 2=Vermelha P1, 3=Vermelha P2, 4=Escassez Hidrica',
# MAGIC     NomBandeiraAcionada STRING COMMENT 'Nome da bandeira acionada no mes (ultima leitura do mes de competencia)',
# MAGIC     ValorAdicionalBandeira DOUBLE COMMENT 'Valor adicional da bandeira no mes, em R$/MWh (ANEEL)',
# MAGIC     EarPercentualNacional DOUBLE COMMENT 'EAR nacional, media simples dos 4 subsistemas, em % da capacidade maxima',
# MAGIC     EnaPercentualMltNacional DOUBLE COMMENT 'ENA nacional, media simples dos 4 subsistemas, em % da MLT (media de longo termo)',
# MAGIC     CmoMedioNacional DOUBLE COMMENT 'CMO nacional, media simples dos 4 subsistemas, em R$/MWh',
# MAGIC     CargaTotalNacional DOUBLE COMMENT 'Carga de energia nacional, soma dos 4 subsistemas no mes, em MWmed',
# MAGIC     DatCarga TIMESTAMP COMMENT 'Data e hora do processamento na camada refined'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Painel mensal analitico consolidado do projeto: primeira consolidacao cross-source da camada refined, cruzando a bandeira tarifaria (ANEEL) com hidrologia (EAR, ENA) e indicadores operacionais (CMO, carga), todos vindos das tabelas trusted. Granularidade: uma linha por mes. Fonte: mba.raw.bandeira_acionada e mba.trusted.f_ear_subsistema / f_ena_subsistema / f_cmo_subsistema / f_carga_energia_subsistema.'

# COMMAND ----------

# DBTITLE 1,Alimenta tabela
# MAGIC %sql
# MAGIC MERGE INTO mba.refined.painel_mensal_bandeira_hidrologia AS tgt
# MAGIC USING (
# MAGIC     WITH bandeira_mes AS (
# MAGIC         SELECT
# MAGIC             DatCompetencia,
# MAGIC             NomBandeiraAcionada,
# MAGIC             VlrAdicionalBandeira,
# MAGIC             CAST(date_format(DatCompetencia, 'yyyyMM') AS INT) AS MesCompetencia,
# MAGIC             date_format(DatCompetencia, 'yyyy-MM') AS AnoMes,
# MAGIC             ROW_NUMBER() OVER (
# MAGIC                 PARTITION BY date_format(DatCompetencia, 'yyyyMM')
# MAGIC                 ORDER BY DatCompetencia DESC
# MAGIC             ) AS rn
# MAGIC         FROM mba.raw.bandeira_acionada
# MAGIC         WHERE DatCompetencia IS NOT NULL
# MAGIC     ),
# MAGIC     bandeira_final AS (
# MAGIC         SELECT
# MAGIC             MesCompetencia,
# MAGIC             AnoMes,
# MAGIC             NomBandeiraAcionada,
# MAGIC             VlrAdicionalBandeira,
# MAGIC             CASE trim(NomBandeiraAcionada)
# MAGIC                 WHEN 'Verde' THEN 0
# MAGIC                 WHEN 'Amarela' THEN 1
# MAGIC                 WHEN 'Vermelha P1' THEN 2
# MAGIC                 WHEN 'Vermelha P2' THEN 3
# MAGIC                 WHEN 'Escassez Hídrica' THEN 4
# MAGIC                 ELSE NULL
# MAGIC             END AS NivelBandeira
# MAGIC         FROM bandeira_mes
# MAGIC         WHERE rn = 1
# MAGIC     ),
# MAGIC     ear_mes AS (
# MAGIC         SELECT
# MAGIC             CAST(date_format(ear_data, 'yyyyMM') AS INT) AS MesCompetencia,
# MAGIC             AVG(ear_verif_subsistema_percentual) AS EarPercentualNacional
# MAGIC         FROM mba.trusted.f_ear_subsistema
# MAGIC         GROUP BY 1
# MAGIC     ),
# MAGIC     ena_mes AS (
# MAGIC         SELECT
# MAGIC             CAST(date_format(ena_data, 'yyyyMM') AS INT) AS MesCompetencia,
# MAGIC             AVG(ena_bruta_regiao_percentualmlt) AS EnaPercentualMltNacional
# MAGIC         FROM mba.trusted.f_ena_subsistema
# MAGIC         GROUP BY 1
# MAGIC     ),
# MAGIC     cmo_mes AS (
# MAGIC         SELECT
# MAGIC             CAST(date_format(din_instante, 'yyyyMM') AS INT) AS MesCompetencia,
# MAGIC             AVG(val_cmomediasemanal) AS CmoMedioNacional
# MAGIC         FROM mba.trusted.f_cmo_subsistema
# MAGIC         GROUP BY 1
# MAGIC     ),
# MAGIC     carga_mes AS (
# MAGIC         SELECT
# MAGIC             CAST(date_format(din_instante, 'yyyyMM') AS INT) AS MesCompetencia,
# MAGIC             SUM(val_cargaenergiamwmed) AS CargaTotalNacional
# MAGIC         FROM mba.trusted.f_carga_energia_subsistema
# MAGIC         GROUP BY 1
# MAGIC     )
# MAGIC     SELECT
# MAGIC         b.MesCompetencia,
# MAGIC         b.AnoMes,
# MAGIC         b.NivelBandeira,
# MAGIC         b.NomBandeiraAcionada,
# MAGIC         b.VlrAdicionalBandeira AS ValorAdicionalBandeira,
# MAGIC         e.EarPercentualNacional,
# MAGIC         n.EnaPercentualMltNacional,
# MAGIC         c.CmoMedioNacional,
# MAGIC         g.CargaTotalNacional,
# MAGIC         current_timestamp() AS DatCarga
# MAGIC     FROM bandeira_final b
# MAGIC     LEFT JOIN ear_mes e ON e.MesCompetencia = b.MesCompetencia
# MAGIC     LEFT JOIN ena_mes n ON n.MesCompetencia = b.MesCompetencia
# MAGIC     LEFT JOIN cmo_mes c ON c.MesCompetencia = b.MesCompetencia
# MAGIC     LEFT JOIN carga_mes g ON g.MesCompetencia = b.MesCompetencia
# MAGIC ) AS src
# MAGIC ON tgt.MesCompetencia = src.MesCompetencia
# MAGIC
# MAGIC WHEN MATCHED THEN
# MAGIC     UPDATE SET
# MAGIC         tgt.AnoMes = src.AnoMes,
# MAGIC         tgt.NivelBandeira = src.NivelBandeira,
# MAGIC         tgt.NomBandeiraAcionada = src.NomBandeiraAcionada,
# MAGIC         tgt.ValorAdicionalBandeira = src.ValorAdicionalBandeira,
# MAGIC         tgt.EarPercentualNacional = src.EarPercentualNacional,
# MAGIC         tgt.EnaPercentualMltNacional = src.EnaPercentualMltNacional,
# MAGIC         tgt.CmoMedioNacional = src.CmoMedioNacional,
# MAGIC         tgt.CargaTotalNacional = src.CargaTotalNacional,
# MAGIC         tgt.DatCarga = src.DatCarga
# MAGIC
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     INSERT (MesCompetencia, AnoMes, NivelBandeira, NomBandeiraAcionada, ValorAdicionalBandeira, EarPercentualNacional, EnaPercentualMltNacional, CmoMedioNacional, CargaTotalNacional, DatCarga)
# MAGIC     VALUES (src.MesCompetencia, src.AnoMes, src.NivelBandeira, src.NomBandeiraAcionada, src.ValorAdicionalBandeira, src.EarPercentualNacional, src.EnaPercentualMltNacional, src.CmoMedioNacional, src.CargaTotalNacional, src.DatCarga);

# COMMAND ----------

# DBTITLE 1,Valida contagem
# MAGIC %sql
# MAGIC SELECT COUNT(*) AS total_linhas, MIN(AnoMes) AS primeiro_mes, MAX(AnoMes) AS ultimo_mes
# MAGIC FROM mba.refined.painel_mensal_bandeira_hidrologia

# COMMAND ----------

dbutils.notebook.exit("OK")
