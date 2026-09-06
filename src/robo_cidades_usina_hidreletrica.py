"""
"Robô" em Python para identificar as cidades brasileiras que possuem usina
hidrelétrica (UHE, PCH ou CGH), a partir da base oficial do SIGA (Sistema de
Informações de Geração da ANEEL).

Fonte oficial (Portal de Dados Abertos da ANEEL):
  https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel

O que o script faz:
  1. Baixa o CSV bruto "siga-empreendimentos-geracao.csv" (todos os
     empreendimentos de geração de energia do Brasil).
  2. Filtra apenas os empreendimentos de fonte hídrica:
       - UHE  -> Usina Hidrelétrica
       - PCH  -> Pequena Central Hidrelétrica
       - CGH  -> Central Geradora Hidrelétrica
  3. Trata a coluna de municípios (uma usina pode abranger mais de um
     município) e "explode" essa lista em uma linha por município.
  4. Gera:
       a) um CSV com o detalhamento de cada usina hidrelétrica por município;
       b) um CSV resumido com a lista única de cidades (município + UF) que
          possuem ao menos uma usina hidrelétrica, com contagem de usinas e
          potência total instalada.

Como executar:
    python robo_cidades_hidreletricas.py
"""

import os
import urllib.request

import numpy as np
import pandas as pd

URL_SIGA = (
    "https://dadosabertos.aneel.gov.br/dataset/6d90b77c-c5f5-4d81-bdec-7bc619494bb9/"
    "resource/11ec447d-698d-4ab8-977f-b424d5deee6a/download/siga-empreendimentos-geracao.csv"
)

PASTA_BRUTOS = "dados_brutos"
PASTA_SAIDA = "dados_tratados"
ARQUIVO_BRUTO = os.path.join(PASTA_BRUTOS, "siga-empreendimentos-geracao.csv")
ARQUIVO_DETALHE = os.path.join(PASTA_SAIDA, "usinas_hidreletricas_por_municipio.csv")
ARQUIVO_CIDADES = os.path.join(PASTA_SAIDA, "cidades_com_usina_hidreletrica.csv")

TIPOS_HIDRICOS = {
    "UHE": "Usina Hidrelétrica",
    "PCH": "Pequena Central Hidrelétrica",
    "CGH": "Central Geradora Hidrelétrica",
}


def baixar_arquivo(url: str, destino: str) -> None:
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if os.path.exists(destino):
        print(f"[ok] já existe localmente: {destino}")
        return
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(destino, "wb") as f:
        f.write(resp.read())
    print(f"[ok] salvo em: {destino}")


def para_numero_br(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan})
        .astype(float)
    )


def carregar_e_filtrar_hidricas() -> pd.DataFrame:
    df = pd.read_csv(ARQUIVO_BRUTO, sep=";", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    # Mantém apenas empreendimentos de fonte hídrica (UHE, PCH, CGH)
    df = df[df["SigTipoGeracao"].isin(TIPOS_HIDRICOS.keys())].copy()

    # Limpeza básica
    df["NomEmpreendimento"] = df["NomEmpreendimento"].astype(str).str.strip()
    df["SigUFPrincipal"] = df["SigUFPrincipal"].astype(str).str.strip()
    df["DscFaseUsina"] = df["DscFaseUsina"].astype(str).str.strip()
    df["DscMuninicpios"] = df["DscMuninicpios"].astype(str).str.strip()
    df["DatEntradaOperacao"] = pd.to_datetime(df["DatEntradaOperacao"], errors="coerce")

    for col in ["MdaPotenciaOutorgadaKw", "MdaPotenciaFiscalizadaKw", "MdaGarantiaFisicaKw"]:
        df[col] = para_numero_br(df[col])
        df[col] = df[col].fillna(0.0)

    df["DscTipoGeracaoExtenso"] = df["SigTipoGeracao"].map(TIPOS_HIDRICOS)

    # Remove registros sem nenhuma informação de município (não é possível localizar a cidade)
    df = df[df["DscMuninicpios"].notna() & (df["DscMuninicpios"] != "") & (df["DscMuninicpios"] != "nan")]

    return df.reset_index(drop=True)


def explodir_por_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """
    A coluna DscMuninicpios pode conter mais de um município, no formato:
    'Município A - UF, Município B - UF'. Esta função cria uma linha por
    município para cada usina.
    """
    registros = []
    for _, linha in df.iterrows():
        municipios_brutos = str(linha["DscMuninicpios"]).split(",")
        for item in municipios_brutos:
            item = item.strip()
            if not item:
                continue
            if " - " in item:
                municipio, uf = item.rsplit(" - ", 1)
            else:
                municipio, uf = item, linha["SigUFPrincipal"]
            registros.append(
                {
                    "Municipio": municipio.strip(),
                    "UF": uf.strip(),
                    "NomEmpreendimento": linha["NomEmpreendimento"],
                    "CodCEG": linha["CodCEG"],
                    "SigTipoGeracao": linha["SigTipoGeracao"],
                    "DscTipoGeracaoExtenso": linha["DscTipoGeracaoExtenso"],
                    "DscFaseUsina": linha["DscFaseUsina"],
                    "DatEntradaOperacao": linha["DatEntradaOperacao"],
                    "MdaPotenciaOutorgadaKw": linha["MdaPotenciaOutorgadaKw"],
                    "MdaPotenciaFiscalizadaKw": linha["MdaPotenciaFiscalizadaKw"],
                    "DscSubBacia": linha.get("DscSubBacia", np.nan),
                }
            )
    detalhado = pd.DataFrame(registros)
    detalhado = detalhado.dropna(subset=["Municipio"])
    detalhado = detalhado.drop_duplicates()
    detalhado["Municipio"] = detalhado["Municipio"].str.strip()
    detalhado["UF"] = detalhado["UF"].str.strip()
    detalhado = detalhado.sort_values(["UF", "Municipio", "NomEmpreendimento"]).reset_index(drop=True)
    return detalhado


def resumir_por_cidade(detalhado: pd.DataFrame) -> pd.DataFrame:
    resumo = (
        detalhado.groupby(["Municipio", "UF"])
        .agg(
            QtdUsinasHidreletricas=("CodCEG", "nunique"),
            QtdUHE=("SigTipoGeracao", lambda s: (s == "UHE").sum()),
            QtdPCH=("SigTipoGeracao", lambda s: (s == "PCH").sum()),
            QtdCGH=("SigTipoGeracao", lambda s: (s == "CGH").sum()),
            QtdEmOperacao=("DscFaseUsina", lambda s: (s == "Operação").sum()),
            PotenciaOutorgadaTotalKw=("MdaPotenciaOutorgadaKw", "sum"),
            NomesUsinas=("NomEmpreendimento", lambda s: "; ".join(sorted(set(s)))),
        )
        .reset_index()
    )
    resumo = resumo.sort_values(["UF", "Municipio"]).reset_index(drop=True)
    return resumo


def main() -> None:
    baixar_arquivo(URL_SIGA, ARQUIVO_BRUTO)

    print("\n[etapa] filtrando empreendimentos de fonte hídrica (UHE, PCH, CGH)...")
    df_hidro = carregar_e_filtrar_hidricas()
    print(f"  -> {df_hidro.shape[0]} usinas hídricas encontradas na base nacional")

    print("\n[etapa] associando cada usina ao(s) seu(s) município(s)...")
    detalhado = explodir_por_municipio(df_hidro)
    print(f"  -> {detalhado.shape[0]} vínculos usina-município")

    print("\n[etapa] resumindo por cidade...")
    resumo = resumir_por_cidade(detalhado)
    print(f"  -> {resumo.shape[0]} cidades brasileiras com ao menos uma usina hidrelétrica")

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    detalhado.to_csv(ARQUIVO_DETALHE, index=False, encoding="utf-8-sig")
    resumo.to_csv(ARQUIVO_CIDADES, index=False, encoding="utf-8-sig")

    print(f"\n[ok] detalhamento usina-município salvo em: {ARQUIVO_DETALHE}")
    print(f"[ok] lista de cidades salva em: {ARQUIVO_CIDADES}")

    print("\n[resumo] Top 10 estados com mais cidades com usina hidrelétrica:")
    print(resumo["UF"].value_counts().head(10))


if __name__ == "__main__":
    main()
