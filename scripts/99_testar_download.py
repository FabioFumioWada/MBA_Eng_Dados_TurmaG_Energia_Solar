from pathlib import Path

import requests


# Arquivo pequeno e público para teste.
URL = (
    "https://dadosabertos.aneel.gov.br/dataset/"
    "7f43a020-6dc5-44b8-80b4-d97eaa94436c/resource/"
    "5879ca80-b3bd-45b1-a135-d9b77c1d5b36/download/"
    "bandeira-tarifaria-adicional.csv"
)

# Execução local:
#DESTINO = Path("teste_bandeira_tarifaria.csv")

# Para executar no Databricks com armazenamento persistente, use, por exemplo:
# DESTINO = Path("/dbfs/FileStore/testes/teste_bandeira_tarifaria.csv")
# ou:

DESTINO = Path("/Volumes/mba/energia/raw/teste_bandeira_tarifaria.csv")


def main() -> None:
    print(f"Iniciando download: {URL}")

    try:
        with requests.get(URL, stream=True, timeout=(15, 60)) as resposta:
            resposta.raise_for_status()
            DESTINO.parent.mkdir(parents=True, exist_ok=True)

            total_bytes = 0
            with DESTINO.open("wb") as arquivo:
                for bloco in resposta.iter_content(chunk_size=64 * 1024):
                    if bloco:
                        arquivo.write(bloco)
                        total_bytes += len(bloco)

        print("Download concluído com sucesso.")
        print(f"Arquivo salvo em: {DESTINO.resolve()}")
        print(f"Tamanho: {total_bytes:,} bytes")

    except requests.RequestException as erro:
        print(f"Erro de conexão ou HTTP: {erro}")
    except OSError as erro:
        print(f"Erro ao gravar o arquivo: {erro}")


if __name__ == "__main__":
    main()
