# Databricks notebook source
# MAGIC %md
# MAGIC ## 04 - EAR Reservatórios (Stage)
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC Baixar os arquivos **CSV anuais** de Energia Armazenada nos Reservatórios (EAR) por
# MAGIC subsistema, direto do portal de Dados Abertos do ONS, e salvar no Volume da camada
# MAGIC `stage`, **sem nenhuma alteração** — exatamente como chegam da fonte (mesmo padrão
# MAGIC usado para os outros datasets desta camada).
# MAGIC
# MAGIC **Fonte:** https://dados.ons.org.br/dataset/ear-diario-por-subsistema
# MAGIC
# MAGIC **Padrão de download:** um arquivo CSV por ano, com o histórico diário de EAR por subsistema (N, NE, S, SE).

# COMMAND ----------

# DBTITLE 1,Configurações
import requests

# ============================================================
# CONFIGURAÇÕES
# ============================================================
ANOS = list(range(2000, 2027))

URL_CSV_ANO = (
    "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/ear_subsistema_di/"
    "EAR_DIARIO_SUBSISTEMA_{ano}.csv"
)

pasta_volume = "/Volumes/mba/stage/dados_bruto/ear"
dbutils.fs.mkdirs(pasta_volume)

# COMMAND ----------

# DBTITLE 1,Download dos arquivos originais (um CSV por ano)
for ano in ANOS:
    url = URL_CSV_ANO.format(ano=ano)
    arquivo_volume = f"{pasta_volume}/ONS_EAR_Diario_Subsistema_{ano}.csv"

    print(f"Baixando EAR {ano} de {url} ...")
    response = requests.get(url, timeout=180)
    response.raise_for_status()

    with open(arquivo_volume, "wb") as arquivo:
        arquivo.write(response.content)

    print(f"  Salvo em {arquivo_volume} ({len(response.content):,} bytes)")

print("\nDownload concluído com sucesso!")

# COMMAND ----------

dbutils.notebook.exit("OK")
