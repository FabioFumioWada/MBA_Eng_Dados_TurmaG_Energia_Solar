# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## 09 - Indice ONI e Previsao ENSO / El Nino-La Nina (Stage)
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC Baixar da NOAA CPC (orgao publico dos Estados Unidos responsavel pelo
# MAGIC monitoramento oficial do fenomeno El Nino/La Nina) dois arquivos de texto
# MAGIC puro (sem imagem, sem PDF) e salvar no Volume da camada `stage`,
# MAGIC praticamente sem alteracao — mesmo padrao usado para os outros datasets
# MAGIC desta camada:
# MAGIC
# MAGIC 1. **Indice ONI historico mensal (1950-presente)** — anomalia de
# MAGIC    temperatura da superficie do Oceano Pacifico (regiao Nino 3.4).
# MAGIC    Fonte: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
# MAGIC 2. **Previsao probabilistica ENSO** (El Nino / La Nina / Neutro) para os
# MAGIC    proximos trimestres moveis, atualizada mensalmente.
# MAGIC    Fonte: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities/
# MAGIC
# MAGIC ### Por que NOAA e nao INMET/CPTEC?
# MAGIC
# MAGIC O INMET e o CPTEC (INPE) divulgam a previsao climatica sazonal brasileira
# MAGIC apenas em boletins com mapas/figuras (formato imagem ou PDF com graficos),
# MAGIC sem uma tabela numerica para download (limitacao ja documentada no
# MAGIC notebook e no relatorio do projeto). O fenomeno El Nino/La Nina em si e
# MAGIC monitorado internacionalmente com base na temperatura do Oceano Pacifico,
# MAGIC e a propria previsao sazonal brasileira (CPTEC/INMET/FUNCEME) se baseia
# MAGIC nesse mesmo monitoramento internacional. Por isso, usar a fonte primaria
# MAGIC (NOAA CPC) e uma pratica valida e padrao em estudos de hidrologia e clima,
# MAGIC inclusive no Brasil.
# MAGIC
# MAGIC **Observacao sobre execucao:** assim como aconteceu com a bandeira
# MAGIC (ANEEL) e o clima (INMET), nao foi possivel confirmar se o dominio
# MAGIC `cpc.ncep.noaa.gov` responde a partir do cluster serverless do Databricks
# MAGIC (o token usado para documentar este notebook nao tem permissao de
# MAGIC `clusters/jobs`, so de `workspace` e `sql`). O codigo abaixo faz a
# MAGIC tentativa de download direto; se o dominio bloquear a requisicao, baixe os
# MAGIC dois arquivos localmente (rodando `robo_noaa_enso.py`) e envie os CSVs
# MAGIC para a pasta do Volume indicada abaixo antes de rodar o notebook seguinte
# MAGIC (`02_processamento_raw/09_oni_enso`). Os dados dessa carga inicial ja
# MAGIC foram gerados e enviados dessa forma manual.

# COMMAND ----------

# DBTITLE 1,Configurações
import re
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0 (trabalho academico MBA Eng Dados)"}
URL_ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
URL_PROB = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities/"

# Convencao oficial NOAA: cada trimestre movel (ex.: "JFM") e atribuido ao mes central
SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

pasta_volume = "/Volumes/mba/stage/dados_bruto/oni_noaa"
dbutils.fs.mkdirs(pasta_volume)


def baixar(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")

# COMMAND ----------

# DBTITLE 1,Indice ONI historico (1950-presente)
import csv

texto = baixar(URL_ONI)
linhas = texto.strip().split("\n")[1:]  # pula cabecalho "SEAS YR TOTAL ANOM"

registros = []
for linha in linhas:
    partes = linha.split()
    if len(partes) != 4:
        continue
    seas, yr, total, anom = partes
    if seas not in SEASON_TO_MONTH:
        continue
    mes = SEASON_TO_MONTH[seas]
    mes_ref = int(yr) * 100 + mes
    registros.append((mes_ref, seas, float(anom)))

registros.sort(key=lambda r: r[0])

caminho_oni = f"{pasta_volume}/oni_historico_mensal.csv"
with open(caminho_oni, "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(["MesRef", "Trimestre", "OniAnomC"])
    escritor.writerows(registros)

print(f"Indice ONI: {len(registros)} linhas salvas em {caminho_oni}")

# COMMAND ----------

# DBTITLE 1,Previsao probabilistica ENSO atual
html = baixar(URL_PROB)
m_emissao = re.search(r"Issued\s+(\w+\s+\d{4})", html)
data_emissao = m_emissao.group(1) if m_emissao else "desconhecida"

linhas_tr = re.findall(r"<tr>(.*?)</tr>", html, re.S)
registros_prev = []
for tr in linhas_tr:
    celulas = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)
    celulas = [re.sub("<[^<]+?>", "", c).strip() for c in celulas]
    if len(celulas) == 4 and re.match(r"^[A-Z]{3}\s", celulas[0]):
        trimestre, la_nina, neutro, el_nino = celulas
        registros_prev.append((data_emissao, trimestre, int(la_nina), int(neutro), int(el_nino)))

caminho_prev = f"{pasta_volume}/previsao_enso_atual.csv"
with open(caminho_prev, "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(["DataEmissao", "Trimestre", "ProbLaNinaPct", "ProbNeutroPct", "ProbElNinoPct"])
    escritor.writerows(registros_prev)

print(f"Previsao ENSO: {len(registros_prev)} linhas salvas em {caminho_prev}")

# COMMAND ----------

dbutils.notebook.exit("OK")