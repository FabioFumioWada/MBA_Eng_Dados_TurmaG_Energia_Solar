# Databricks notebook source
# MAGIC %md
# MAGIC ## Base para o Modelo - Bandeira Tarifária x Clima (Usinas Hidrelétricas)
# MAGIC
# MAGIC ### Objetivo
# MAGIC Montar a tabela final pronta para treinar o modelo (v1) definida na
# MAGIC estratégia definida pelo grupo: usar apenas hidrelétrica + chuva para prever a
# MAGIC bandeira tarifária, deixando EAR, ENA e CMO como enriquecimento futuro.
# MAGIC
# MAGIC ### Cuidado com vazamento de dados (data leakage)
# MAGIC A ANEEL decide a bandeira de um mês ANTES desse mês comecar, olhando
# MAGIC para o clima e a hidrologia de meses anteriores. Por isso, o clima usado
# MAGIC como preditor aqui e sempre do mês ANTERIOR (M-1) ao mês da bandeira
# MAGIC prevista (M). Sem essa defasagem, o modelo pareceria otimo no treino mas
# MAGIC nao serviria para prever nada de verdade (estaria usando informacao que
# MAGIC ainda nao existiria no momento da decisao real).
# MAGIC
# MAGIC ### Sazonalidade (chuva x normal historica do mes)
# MAGIC Comparar milimetros de chuva bruta entre meses diferentes engana, porque
# MAGIC cada mes do ano ja tem um padrao esperado (janeiro e naturalmente mais
# MAGIC chuvoso que agosto no Brasil). O ONS resolve isso comparando a Energia
# MAGIC Natural Afluente (ENA) com a Media de Longo Termo (MLT): diz que um mes
# MAGIC esta "em X% da MLT" em vez de citar o valor bruto. Aplicamos a mesma
# MAGIC logica aqui: `PrecipitacaoPctNormal` compara a chuva do mes de
# MAGIC referencia com a media historica de todos os mesmos meses do ano na
# MAGIC nossa serie (ex.: chuva de janeiro/2016 vs a media de todos os
# MAGIC janeiros). Tambem incluimos `Mes` (1 a 12) como feature, para que o
# MAGIC modelo aprenda o padrao sazonal diretamente.
# MAGIC
# MAGIC ### Granularidade
# MAGIC Uma linha por mês alvo (MesCompetencia, formato YYYYMM), com o clima do
# MAGIC mês anterior (MesReferenciaClima).
# MAGIC
# MAGIC ### Como as fontes se conectam
# MAGIC 1. `mba.trusted.d_usina_estacao` -> lista as estações meteorológicas
# MAGIC    ligadas a cada usina hidrelétrica (casamento por município, com
# MAGIC    fallback por distância geográfica - notebook 04 da camada trusted).
# MAGIC 2. `mba.trusted.f_clima_diario` -> clima diário dessas estações
# MAGIC    (INMET), agregado para o mês (chuva acumulada, chuva média,
# MAGIC    temperatura média, umidade média).
# MAGIC 3. `mba.trusted.f_bandeira` -> bandeira tarifária do mesmo mês (ANEEL),
# MAGIC    usada como rótulo (`IsVermelha`: 1 = vermelha, 0 = demais).

# COMMAND ----------

# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mba.refined.f_modelo_bandeira_clima (
# MAGIC     MesCompetencia INT COMMENT 'Mês em que a bandeira foi acionada (mês alvo, o que o modelo tenta prever), formato YYYYMM',
# MAGIC     MesReferenciaClima INT COMMENT 'Mês do clima usado como preditor: sempre o mês anterior ao MesCompetencia, para evitar vazamento de dados',
# MAGIC     Mes INT COMMENT 'Mês do ano (1 a 12) do clima de referência, usado para o modelo aprender o padrão sazonal',
# MAGIC     QtdEstacoesUsadas INT COMMENT 'Quantidade de estações meteorológicas ligadas a usinas usadas na média do mês de referência',
# MAGIC     PrecipitacaoMediaMm DOUBLE COMMENT 'Precipitação média diária do mês, em mm, entre as estações ligadas a usinas',
# MAGIC     PrecipitacaoAcumuladaMm DOUBLE COMMENT 'Precipitação acumulada no mês, em mm, somando todas as estações ligadas a usinas',
# MAGIC     PrecipitacaoNormalMm DOUBLE COMMENT 'Normal histórica de chuva para esse mês do ano (média de todos os anos da série para o mesmo mês)',
# MAGIC     PrecipitacaoPctNormal DOUBLE COMMENT 'Percentual da chuva acumulada em relação a normal histórica do mês (mesma lógica do indicador ENA em %MLT usado pelo ONS)',
# MAGIC     TemperaturaMediaC DOUBLE COMMENT 'Temperatura média do mês, em °C, entre as estações ligadas a usinas',
# MAGIC     UmidadeMediaPct DOUBLE COMMENT 'Umidade relativa média do mês, em %, entre as estações ligadas a usinas',
# MAGIC     IsVermelha TINYINT COMMENT 'Rótulo do modelo: 1 = bandeira vermelha acionada no mês; 0 = demais bandeiras',
# MAGIC     DatCarga TIMESTAMP COMMENT 'Data e hora do processamento na camada refined'
# MAGIC )
# MAGIC USING DELTA
# MAGIC COMMENT 'Tabela pronta para treinar o modelo v1 (Sprint 4): clima das estações ligadas a usinas hidrelétricas x bandeira tarifária do mês. Fontes: INMET (via mba.trusted.f_clima_diario) e ANEEL (via mba.trusted.f_bandeira).';

# COMMAND ----------

# DBTITLE 1,Agrega clima mensal das estações ligadas a usinas
from pyspark.sql import functions as F

spark.sql("""
    WITH usina_estacoes AS (
        SELECT DISTINCT cod_estacao
        FROM mba.trusted.d_usina_estacao
        WHERE cod_estacao IS NOT NULL
    ),
    clima_usinas AS (
        SELECT c.*, CAST(date_format(c.data, 'yyyyMM') AS INT) AS MesReferenciaClima
        FROM mba.trusted.f_clima_diario c
        JOIN usina_estacoes u ON c.codigo_wmo = u.cod_estacao
    ),
    clima_mensal AS (
        SELECT
            MesReferenciaClima,
            CAST(SUBSTRING(CAST(MesReferenciaClima AS STRING), 5, 2) AS INT) AS Mes,
            COUNT(DISTINCT codigo_wmo) AS QtdEstacoesUsadas,
            ROUND(AVG(precipitacao_total_dia_mm), 2) AS PrecipitacaoMediaMm,
            ROUND(SUM(precipitacao_total_dia_mm), 2) AS PrecipitacaoAcumuladaMm,
            ROUND(AVG(temperatura_media_c), 2) AS TemperaturaMediaC,
            ROUND(AVG(umidade_relativa_media_pct), 2) AS UmidadeMediaPct
        FROM clima_usinas
        GROUP BY MesReferenciaClima
    ),
    normal_mensal AS (
        SELECT Mes, ROUND(AVG(PrecipitacaoAcumuladaMm), 2) AS PrecipitacaoNormalMm
        FROM clima_mensal
        GROUP BY Mes
    )
    SELECT
        m.MesReferenciaClima, m.Mes, m.QtdEstacoesUsadas, m.PrecipitacaoMediaMm,
        m.PrecipitacaoAcumuladaMm, n.PrecipitacaoNormalMm,
        ROUND(m.PrecipitacaoAcumuladaMm / n.PrecipitacaoNormalMm * 100, 1) AS PrecipitacaoPctNormal,
        m.TemperaturaMediaC, m.UmidadeMediaPct
    FROM clima_mensal m
    JOIN normal_mensal n ON m.Mes = n.Mes
""").createOrReplaceTempView("stg_clima_mensal_usinas")

# COMMAND ----------

# DBTITLE 1,Junta com a bandeira do mês e grava (idempotente)
# MAGIC %sql
# MAGIC MERGE INTO mba.refined.f_modelo_bandeira_clima AS tgt
# MAGIC USING (
# MAGIC     SELECT
# MAGIC         b.MesCompetencia,
# MAGIC         m.MesReferenciaClima,
# MAGIC         m.Mes,
# MAGIC         m.QtdEstacoesUsadas,
# MAGIC         m.PrecipitacaoMediaMm,
# MAGIC         m.PrecipitacaoAcumuladaMm,
# MAGIC         m.PrecipitacaoNormalMm,
# MAGIC         m.PrecipitacaoPctNormal,
# MAGIC         m.TemperaturaMediaC,
# MAGIC         m.UmidadeMediaPct,
# MAGIC         b.IsVermelha,
# MAGIC         current_timestamp() AS DatCarga
# MAGIC     FROM mba.trusted.f_bandeira b
# MAGIC     JOIN stg_clima_mensal_usinas m
# MAGIC       ON m.MesReferenciaClima = CAST(date_format(add_months(to_date(CAST(b.MesCompetencia AS STRING), 'yyyyMM'), -1), 'yyyyMM') AS INT)
# MAGIC ) AS src
# MAGIC ON tgt.MesCompetencia = src.MesCompetencia
# MAGIC WHEN MATCHED THEN UPDATE SET *
# MAGIC WHEN NOT MATCHED THEN INSERT *;

# COMMAND ----------

# DBTITLE 1,Confere o resultado
# MAGIC %sql
# MAGIC SELECT COUNT(*) AS total_meses, SUM(IsVermelha) AS meses_bandeira_vermelha,
# MAGIC        MIN(MesCompetencia) AS primeiro_mes, MAX(MesCompetencia) AS ultimo_mes
# MAGIC FROM mba.refined.f_modelo_bandeira_clima;

# COMMAND ----------

dbutils.notebook.exit("Executed")
