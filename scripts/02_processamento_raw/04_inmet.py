# Databricks notebook source
# DBTITLE 1,Cria tabela
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- TABELA: INMET - DADOS HISTÓRICOS HORÁRIOS POR ESTAÇÃO
# MAGIC -- ORIGEM: INMET - Instituto Nacional de Meteorologia (BDMEP)
# MAGIC -- DATASET: dadoshistoricos ({ano}.zip, um CSV por estação)
# MAGIC -- FORMATO: Delta Lake
# MAGIC -- FREQUÊNCIA DE ATUALIZAÇÃO: Anual (arquivo fechado por ano)
# MAGIC -- ============================================================
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS mba.raw.clima_inmet
# MAGIC (
# MAGIC     -- ========================================================
# MAGIC     -- IDENTIFICAÇÃO DA ESTAÇÃO (metadados do cabeçalho do CSV)
# MAGIC     -- ========================================================
# MAGIC     regiao STRING
# MAGIC         COMMENT 'Região do Brasil onde está a estação.',
# MAGIC     uf STRING
# MAGIC         COMMENT 'UF onde está a estação.',
# MAGIC     estacao STRING
# MAGIC         COMMENT 'Nome da estação meteorológica.',
# MAGIC     codigo_wmo STRING
# MAGIC         COMMENT 'Código OMM/WMO da estação (identificador único).',
# MAGIC     latitude DOUBLE
# MAGIC         COMMENT 'Latitude da estação, em grau decimal.',
# MAGIC     longitude DOUBLE
# MAGIC         COMMENT 'Longitude da estação, em grau decimal.',
# MAGIC     altitude_m DOUBLE
# MAGIC         COMMENT 'Altitude da estação, em metros.',
# MAGIC     data_fundacao STRING
# MAGIC         COMMENT 'Data de fundação da estação, como veio na fonte.',
# MAGIC     ano INT
# MAGIC         COMMENT 'Ano de referência do arquivo de origem.',
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- LEITURA HORÁRIA
# MAGIC     -- ========================================================
# MAGIC     data DATE
# MAGIC         COMMENT 'Data da observação (YYYY-MM-DD).',
# MAGIC     hora_utc STRING
# MAGIC         COMMENT 'Hora da observação em UTC, como veio na fonte.',
# MAGIC     precipitacao_total_mm DOUBLE
# MAGIC         COMMENT 'Precipitação total, horária, em mm.',
# MAGIC     pressao_atm_mb DOUBLE
# MAGIC         COMMENT 'Pressão atmosférica ao nível da estação, horária, em mB.',
# MAGIC     pressao_atm_max_mb DOUBLE
# MAGIC         COMMENT 'Pressão atmosférica máxima na hora anterior, em mB.',
# MAGIC     pressao_atm_min_mb DOUBLE
# MAGIC         COMMENT 'Pressão atmosférica mínima na hora anterior, em mB.',
# MAGIC     radiacao_global_kj_m2 DOUBLE
# MAGIC         COMMENT 'Radiação global, em Kj/m².',
# MAGIC     temperatura_ar_c DOUBLE
# MAGIC         COMMENT 'Temperatura do ar (bulbo seco), horária, em °C.',
# MAGIC     temperatura_orvalho_c DOUBLE
# MAGIC         COMMENT 'Temperatura do ponto de orvalho, horária, em °C.',
# MAGIC     temperatura_max_c DOUBLE
# MAGIC         COMMENT 'Temperatura máxima na hora anterior, em °C.',
# MAGIC     temperatura_min_c DOUBLE
# MAGIC         COMMENT 'Temperatura mínima na hora anterior, em °C.',
# MAGIC     temperatura_orvalho_max_c DOUBLE
# MAGIC         COMMENT 'Temperatura orvalho máxima na hora anterior, em °C.',
# MAGIC     temperatura_orvalho_min_c DOUBLE
# MAGIC         COMMENT 'Temperatura orvalho mínima na hora anterior, em °C.',
# MAGIC     umidade_max_pct DOUBLE
# MAGIC         COMMENT 'Umidade relativa máxima na hora anterior, em %.',
# MAGIC     umidade_min_pct DOUBLE
# MAGIC         COMMENT 'Umidade relativa mínima na hora anterior, em %.',
# MAGIC     umidade_relativa_pct DOUBLE
# MAGIC         COMMENT 'Umidade relativa do ar, horária, em %.',
# MAGIC     vento_direcao_gr DOUBLE
# MAGIC         COMMENT 'Direção do vento, horária, em graus (0-360).',
# MAGIC     vento_rajada_ms DOUBLE
# MAGIC         COMMENT 'Rajada máxima de vento, horária, em m/s.',
# MAGIC     vento_velocidade_ms DOUBLE
# MAGIC         COMMENT 'Velocidade horária do vento, em m/s.',
# MAGIC
# MAGIC     -- ========================================================
# MAGIC     -- METADADOS TÉCNICOS DA INGESTÃO
# MAGIC     -- ========================================================
# MAGIC     NmArquivoCarga STRING
# MAGIC         COMMENT 'Nome do arquivo CSV de origem (uma estação/ano) utilizado na carga.',
# MAGIC     DatCarga TIMESTAMP
# MAGIC         COMMENT 'Data e hora em que o registro foi carregado na tabela Delta.'
# MAGIC )
# MAGIC
# MAGIC USING DELTA
# MAGIC
# MAGIC COMMENT 'Leituras horárias por estação meteorológica do INMET (BDMEP), sem agregação. Valor sentinela -9999 da fonte já convertido para NULL. Fonte: INMET - Instituto Nacional de Meteorologia. Dataset: dadoshistoricos.';

# COMMAND ----------

# DBTITLE 1,Configurações e funções de leitura
import glob
import os
import zipfile

import numpy as np
import pandas as pd

ANOS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
# Extensão de profundidade histórica (2026-09): processamos aqui só os anos novos.
# 2024-2026 já foram carregados antes e não precisam ser reprocessados.
PASTA_VOLUME = "/Volumes/mba/stage/dados_bruto/inmet"
PASTA_EXTRACAO = "/local_disk0/tmp/inmet_extraido"

# Nomes fixos das 19 colunas de dados horárias (o cabeçalho original do INMET
# vem com problema de codificação, por isso usamos nomes fixos e conhecidos,
# na mesma ordem documentada pelo próprio INMET).
COLUNAS_HORARIAS = [
    "data",
    "hora_utc",
    "precipitacao_total_mm",
    "pressao_atm_mb",
    "pressao_atm_max_mb",
    "pressao_atm_min_mb",
    "radiacao_global_kj_m2",
    "temperatura_ar_c",
    "temperatura_orvalho_c",
    "temperatura_max_c",
    "temperatura_min_c",
    "temperatura_orvalho_max_c",
    "temperatura_orvalho_min_c",
    "umidade_max_pct",
    "umidade_min_pct",
    "umidade_relativa_pct",
    "vento_direcao_gr",
    "vento_rajada_ms",
    "vento_velocidade_ms",
]


def para_numero_br(valor):
    """Converte string BR (vírgula decimal) em float, tratando -9999 como nulo."""
    if valor is None:
        return np.nan
    texto = str(valor).strip()
    if texto in ("", "-9999", "-9999,0", "-9999.0"):
        return np.nan
    try:
        return float(texto.replace(".", "").replace(",", "."))
    except ValueError:
        return np.nan


def ler_metadados_estacao(caminho):
    with open(caminho, encoding="latin-1") as f:
        linhas_cabecalho = [next(f) for _ in range(8)]

    def valor(linha):
        return linha.split(";", 1)[1].strip()

    return {
        "regiao": valor(linhas_cabecalho[0]),
        "uf": valor(linhas_cabecalho[1]),
        "estacao": valor(linhas_cabecalho[2]),
        "codigo_wmo": valor(linhas_cabecalho[3]),
        "latitude": para_numero_br(valor(linhas_cabecalho[4])),
        "longitude": para_numero_br(valor(linhas_cabecalho[5])),
        "altitude_m": para_numero_br(valor(linhas_cabecalho[6])),
        "data_fundacao": valor(linhas_cabecalho[7]),
    }


def ler_dados_horarios(caminho):
    return pd.read_csv(
        caminho,
        sep=";",
        header=None,
        names=COLUNAS_HORARIAS,
        skiprows=9,  # 8 linhas de metadados + 1 linha de cabeçalho das colunas
        usecols=range(19),
        encoding="latin-1",
        na_values=["-9999", "-9999,0", ""],
        decimal=",",
        engine="python",
        on_bad_lines="skip",
    )

# COMMAND ----------

# DBTITLE 1,Extrai os ZIPs da camada stage e carrega ano a ano
from pyspark.sql import functions as F

# Pasta de staging dentro do Volume (visível tanto pelo driver/pandas quanto
# pelo cluster Spark). Usamos parquet como "ponte" em vez de
# spark.createDataFrame(pandas_df) porque, no serverless (Spark Connect), o
# createDataFrame precisa transportar todas as linhas via rede (RPC) do
# processo do notebook até o runtime remoto — com milhões de linhas isso
# estoura o tempo limite da chamada (DEADLINE_EXCEEDED). Gravando primeiro em
# parquet dentro do Volume e depois lendo com spark.read.parquet, a leitura é
# feita direto no armazenamento pelo cluster, sem esse gargalo de rede.
PASTA_STAGING_PARQUET = "/Volumes/mba/stage/dados_bruto/_tmp_parquet_clima_inmet"

primeiro_ano = True

for ano in ANOS:
    zip_path = f"{PASTA_VOLUME}/{ano}.zip"
    pasta_extraida = f"{PASTA_EXTRACAO}/{ano}"
    os.makedirs(pasta_extraida, exist_ok=True)

    print(f"Extraindo {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(pasta_extraida)

    arquivos = sorted(
        glob.glob(os.path.join(pasta_extraida, "**", "*.CSV"), recursive=True)
        + glob.glob(os.path.join(pasta_extraida, "**", "*.csv"), recursive=True)
    )
    print(f"Ano {ano}: {len(arquivos)} arquivos de estação encontrados")

    blocos_ano = []
    total_com_erro = 0
    for caminho in arquivos:
        try:
            meta = ler_metadados_estacao(caminho)
            df_horario = ler_dados_horarios(caminho)
            df_horario["data"] = pd.to_datetime(df_horario["data"], format="%Y/%m/%d", errors="coerce").dt.date
            df_horario["ano"] = ano
            for campo, val in meta.items():
                df_horario[campo] = val
            df_horario["NmArquivoCarga"] = os.path.basename(caminho)
            blocos_ano.append(df_horario)
        except Exception as exc:
            total_com_erro += 1
            print(f"  [erro] {os.path.basename(caminho)}: {exc}")

    if not blocos_ano:
        print(f"Ano {ano}: nenhum arquivo processado com sucesso, pulando gravação.")
        continue

    consolidado_ano = pd.concat(blocos_ano, ignore_index=True)
    for coluna in ["regiao", "uf", "estacao", "codigo_wmo"]:
        consolidado_ano[coluna] = consolidado_ano[coluna].astype(str).str.strip()

    # Garante tipos consistentes já no pandas (evita ambiguidade de schema
    # quando o Spark ler o parquet de volta).
    consolidado_ano["ano"] = consolidado_ano["ano"].astype("int32")
    consolidado_ano["latitude"] = consolidado_ano["latitude"].astype("float64")
    consolidado_ano["longitude"] = consolidado_ano["longitude"].astype("float64")
    consolidado_ano["altitude_m"] = consolidado_ano["altitude_m"].astype("float64")

    # Grava em parquet dentro do Volume (I/O local ao driver, rápido) em vez
    # de mandar o DataFrame inteiro pela rede via spark.createDataFrame.
    staging_path = f"{PASTA_STAGING_PARQUET}/ano={ano}"
    os.makedirs(staging_path, exist_ok=True)
    consolidado_ano.to_parquet(f"{staging_path}/part.parquet", index=False, engine="pyarrow")

    # Agora sim o Spark lê o parquet direto do armazenamento (processamento
    # feito no cluster remoto, sem gargalo de rede) e grava na tabela Delta.
    df_spark = (
        spark.read.parquet(staging_path)
        .withColumn("ano", F.col("ano").cast("int"))
        .withColumn("data", F.col("data").cast("date"))
        .withColumn("latitude", F.col("latitude").cast("double"))
        .withColumn("longitude", F.col("longitude").cast("double"))
        .withColumn("altitude_m", F.col("altitude_m").cast("double"))
        .withColumn("DatCarga", F.current_timestamp())
    )

    # A tabela já existe com os anos 2024-2026 carregados antes; sempre "append" aqui
    # para não sobrescrever o que já está na tabela.
    modo = "append"
    (
        df_spark.write
        .format("delta")
        .mode(modo)
        .option("mergeSchema", "true")
        .saveAsTable("mba.raw.clima_inmet")
    )
    primeiro_ano = False

    # Limpa o parquet temporário do ano (já está gravado na Delta table).
    for arq in glob.glob(f"{staging_path}/*"):
        os.remove(arq)
    os.rmdir(staging_path)

    print(f"Ano {ano}: {len(consolidado_ano):,} linhas gravadas em mba.raw.clima_inmet "
          f"(arquivos com erro: {total_com_erro})")

print("\nCarga da camada raw concluída!")

# COMMAND ----------

dbutils.notebook.exit("OK")

# COMMAND ----------

display(
    spark.sql("""
        SELECT *
        FROM mba.raw.clima_inmet
        LIMIT 20
    """)
)

# Quantidade de registros
spark.sql("""
    SELECT ano, COUNT(*) AS quantidade_registros, COUNT(DISTINCT codigo_wmo) AS estacoes
    FROM mba.raw.clima_inmet
    GROUP BY ano
    ORDER BY ano
""").show()
