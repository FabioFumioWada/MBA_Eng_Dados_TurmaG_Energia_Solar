# Databricks notebook source
# MAGIC %md
# MAGIC ## 03 - INMET (Stage)
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC Baixar os arquivos **ZIP anuais** do INMET (Instituto Nacional de Meteorologia) diretamente
# MAGIC do portal oficial e salvar no Volume da camada `stage`, **sem nenhuma alteração** exatamente
# MAGIC como chegam da fonte (mesmo padrão usado para os outros datasets desta camada).
# MAGIC
# MAGIC **Fonte:** https://portal.inmet.gov.br/dadoshistoricos
# MAGIC
# MAGIC **Padrão de download:** um arquivo ZIP por ano, contendo um CSV por estação meteorológica do Brasil.
# MAGIC
# MAGIC ### Observação técnica
# MAGIC
# MAGIC O servidor do INMET derruba a conexão quando o pedido não tem um cabeçalho de `User-Agent`
# MAGIC de navegador, por isso simulamos aqui um pedido normal de navegador, igual ao que acontece
# MAGIC quando você clica no link do site.

# COMMAND ----------

# DBTITLE 1,Configurações
import requests

# ============================================================
# CONFIGURAÇÕES
# ============================================================
ANOS = [2024, 2025, 2026]

URL_ZIP_ANO = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip"

# Precisa parecer uma requisição de navegador de verdade — o servidor do INMET
# derruba a conexão em pedidos sem esse cabeçalho.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

pasta_volume = "/Volumes/mba/stage/dados_bruto/inmet"
dbutils.fs.mkdirs(pasta_volume)

# COMMAND ----------

# DBTITLE 1,Download dos arquivos originais (um ZIP por ano)
for ano in ANOS:
    url = URL_ZIP_ANO.format(ano=ano)
    arquivo_volume = f"{pasta_volume}/{ano}.zip"

    print(f"Baixando clima INMET {ano} de {url} ...")
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
    response.raise_for_status()

    with open(arquivo_volume, "wb") as arquivo:
        arquivo.write(response.content)

    print(f"  Salvo em {arquivo_volume} ({len(response.content):,} bytes)")

print("\nDownload concluído com sucesso!")

# COMMAND ----------

dbutils.notebook.exit("OK")