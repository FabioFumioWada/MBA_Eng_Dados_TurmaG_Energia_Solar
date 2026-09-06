"""
"Robô" em Python para identificar o município de cada estação meteorológica
do INMET (automática ou convencional), a partir do catálogo oficial de
estações da própria INMET.

Fonte oficial (API pública do INMET, mesma usada pelo portal BDMEP):
  https://apitempo.inmet.gov.br/estacoes/T   (estações automáticas)
  https://apitempo.inmet.gov.br/estacoes/M   (estações convencionais)

Por que usar o catálogo oficial em vez de geocodificação por lat/long?
  O campo "DC_NOME" do catálogo do INMET já é o nome do município onde a
  estação está instalada (ex.: "ACAJUTIBA", "ALTAMIRA"), e "SG_ESTADO" é a
  UF. Ou seja, o município vem pronto, direto da fonte oficial, sem precisar
  fazer nenhuma chamada de geocodificação reversa (que depende de APIs
  externas como o Nominatim, tem limite de requisições e falha bastante
  para coordenadas em área rural).

O que o script faz:
  1. Baixa o catálogo de estações automáticas e o de convencionais.
  2. Junta os dois catálogos em uma única tabela.
  3. Renomeia e organiza as colunas (código da estação, nome/município, UF,
     latitude, longitude, altitude, tipo, situação, data de início).
  4. Gera um CSV com o detalhamento de cada estação meteorológica e seu
     município.

Como executar:
    python robo_estacoes_meteorologicas.py
"""

import os
import json
import urllib.request

import pandas as pd

URL_ESTACOES_AUTOMATICAS = "https://apitempo.inmet.gov.br/estacoes/T"
URL_ESTACOES_CONVENCIONAIS = "https://apitempo.inmet.gov.br/estacoes/M"

PASTA_SAIDA = "dados_tratados"
ARQUIVO_DETALHE = os.path.join(PASTA_SAIDA, "estacoes_meteorologicas_por_municipio.csv")

TIPOS_ESTACAO = {
    "Automatica": "T",
    "Convencional": "M",
}


def baixar_json(url: str) -> list:
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dados = json.loads(resp.read().decode("utf-8"))
    print(f"[ok] {len(dados)} estações recebidas")
    return dados


def montar_tabela(dados_automaticas: list, dados_convencionais: list) -> pd.DataFrame:
    df_auto = pd.DataFrame(dados_automaticas)
    df_conv = pd.DataFrame(dados_convencionais)
    df = pd.concat([df_auto, df_conv], ignore_index=True)

    df = df.rename(
        columns={
            "CD_ESTACAO": "CodEstacao",
            "DC_NOME": "Municipio",
            "SG_ESTADO": "UF",
            "TP_ESTACAO": "TipoEstacao",
            "CD_SITUACAO": "Situacao",
            "VL_LATITUDE": "Latitude",
            "VL_LONGITUDE": "Longitude",
            "VL_ALTITUDE": "AltitudeM",
            "DT_INICIO_OPERACAO": "DataInicioOperacao",
            "DT_FIM_OPERACAO": "DataFimOperacao",
        }
    )

    colunas = [
        "CodEstacao",
        "Municipio",
        "UF",
        "TipoEstacao",
        "Situacao",
        "Latitude",
        "Longitude",
        "AltitudeM",
        "DataInicioOperacao",
        "DataFimOperacao",
    ]
    df = df[colunas].copy()

    df["Municipio"] = df["Municipio"].astype(str).str.strip()
    df["UF"] = df["UF"].astype(str).str.strip()
    for col in ["Latitude", "Longitude", "AltitudeM"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["UF", "Municipio", "CodEstacao"]).reset_index(drop=True)
    return df


def main() -> None:
    dados_automaticas = baixar_json(URL_ESTACOES_AUTOMATICAS)
    dados_convencionais = baixar_json(URL_ESTACOES_CONVENCIONAIS)

    print("\n[etapa] montando tabela única de estações com município e UF...")
    detalhado = montar_tabela(dados_automaticas, dados_convencionais)
    print(f"  -> {detalhado.shape[0]} estações meteorológicas identificadas")

    nulos = detalhado["Municipio"].isin(["", "None", "nan"]).sum()
    print(f"  -> {nulos} estações sem município identificado")

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    detalhado.to_csv(ARQUIVO_DETALHE, index=False, encoding="utf-8-sig")
    print(f"\n[ok] detalhamento estação-município salvo em: {ARQUIVO_DETALHE}")

    print("\n[resumo] Top 10 estados com mais estações meteorológicas:")
    print(detalhado["UF"].value_counts().head(10))


if __name__ == "__main__":
    main()
