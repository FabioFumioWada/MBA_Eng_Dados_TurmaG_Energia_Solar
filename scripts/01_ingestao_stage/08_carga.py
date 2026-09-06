# Databricks notebook source
# MAGIC %md
# MAGIC ## 08 - Carga de Energia (Stage)
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC Baixar os arquivos **CSV anuais** de Carga de Energia diária por subsistema, direto do
# MAGIC portal de Dados Abertos do ONS, e salvar no Volume da camada `stage`, **sem nenhuma
# MAGIC alteração** exatamente como chegam da fonte (mesmo padrão usado para os outros
# MAGIC datasets desta camada).
# MAGIC
# MAGIC **Fonte:** https://dados.ons.org.br/dataset/carga-energia-di
# MAGIC
# MAGIC **Padrão de download:** um arquivo CSV por ano, com o histórico diário de carga de energia por subsistema (N, NE, S, SE).

# COMMAND ----------

# DBTITLE 1,Configurações
import requests

# ============================================================
# CONFIGURAÇÕES
# ============================================================
ANOS = list(range(2000, 2027))

URL_CSV_ANO = (
    "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/"
    "CARGA_ENERGIA_{ano}.csv"
)

pasta_volume = "/Volumes/mba/stage/dados_bruto/carga"
dbutils.fs.mkdirs(pasta_volume)

# COMMAND ----------

# DBTITLE 1,Download dos arquivos originais (um CSV por ano)
for ano in ANOS:
    url = URL_CSV_ANO.format(ano=ano)
    arquivo_volume = f"{pasta_volume}/ONS_Carga_Energia_Diaria_{ano}.csv"

    print(f"Baixando Carga de Energia {ano} de {url} ...")
    response = requests.get(url, timeout=180)
    response.raise_for_status()

    with open(arquivo_volume, "wb") as arquivo:
        arquivo.write(response.content)

    print(f"  Salvo em {arquivo_volume} ({len(response.content):,} bytes)")

print("\nDownload concluído com sucesso!")

# COMMAND ----------

dbutils.notebook.exit("OK")