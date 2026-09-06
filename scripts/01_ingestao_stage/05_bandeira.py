# Databricks notebook source
# MAGIC %md
# MAGIC ## 05 - Bandeira Tarifária (Stage)
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC Baixar o arquivo CSV de acionamento da Bandeira Tarifária, direto do Portal de Dados
# MAGIC Abertos da ANEEL, e salvar no Volume da camada `stage`, **sem nenhuma alteração** —
# MAGIC exatamente como chega da fonte (mesmo padrão usado para os outros datasets desta camada).
# MAGIC
# MAGIC **Fonte:** https://dadosabertos.aneel.gov.br/dataset/bandeiras-tarifarias
# MAGIC
# MAGIC **Padrão de download:** um único CSV com o histórico completo de bandeiras acionadas por mês de competência.

# COMMAND ----------

# DBTITLE 1,Configurações
import requests

# ============================================================
# CONFIGURAÇÕES
# ============================================================
URL_ACIONAMENTO = (
    "https://dadosabertos.aneel.gov.br/dataset/7f43a020-6dc5-44b8-80b4-d97eaa94436c/"
    "resource/0591b8f6-fe54-437b-b72b-1aa2efd46e42/download/bandeira-tarifaria-acionamento.csv"
)

pasta_volume = "/Volumes/mba/stage/dados_bruto/ANEEL"
dbutils.fs.mkdirs(pasta_volume)

# COMMAND ----------

# DBTITLE 1,Download do arquivo original
arquivo_volume = f"{pasta_volume}/ANEEL-bandeira-tarifaria-acionamento.csv"

print(f"Baixando Bandeira Tarifária - Acionamento de {URL_ACIONAMENTO} ...")
response = requests.get(URL_ACIONAMENTO, timeout=180)
response.raise_for_status()

with open(arquivo_volume, "wb") as arquivo:
    arquivo.write(response.content)

print(f"  Salvo em {arquivo_volume} ({len(response.content):,} bytes)")
print("\nDownload concluído com sucesso!")

# COMMAND ----------

dbutils.notebook.exit("OK")