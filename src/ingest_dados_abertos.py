#!/usr/bin/env python3
"""Baixa arquivos públicos da ANEEL, ONS e IBGE para um Volume Databricks.

O script descobre os recursos dos conjuntos CKAN da ANEEL e do ONS por meio
 da API package_show e percorre recursivamente o diretório FTP público do IBGE.
Cada arquivo é salvo com o prefixo da plataforma, por exemplo:

    ANEEL-bandeira-tarifaria-adicional.csv
    ONS-ENA_DIARIO_RESERVATORIOS_2026.csv
    IBGE-estimativa_dou_2026.xlsx

A execução é sequencial e possui limitador de taxa, jitter, pausas periódicas,
retries com backoff exponencial, download para arquivo temporário e manifesto
local para rastreabilidade. A raiz do Volume contém as subpastas ANEEL, ONS e
IBGE, criadas automaticamente. Não depende de Spark nem de bibliotecas externas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen


DEFAULT_OUTPUT_DIR = "/Volumes/mba/stage/dados_bruto/"
PLATFORM_OUTPUT_DIRS = {
    "ANEEL": "ANEEL",
    "ONS": "ONS",
    "IBGE": "IBGE",
}
DEFAULT_USER_AGENT = "MBA-Energia-Solar-DadosAbertos/1.0"

ANEEL_DATASET_URL = "https://dadosabertos.aneel.gov.br/dataset/bandeiras-tarifarias"
ONS_DATASET_URL = "https://dados.ons.org.br/dataset/ena-diario-por-reservatorio"
IBGE_ROOT_URL = "https://ftp.ibge.gov.br/Estimativas_de_Populacao/"

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class RemoteFile:
    """Representa um arquivo remoto a ser baixado."""

    platform: str
    url: str
    original_name: str
    resolved_target_name: str | None = None

    @property
    def target_name(self) -> str:
        return self.resolved_target_name or f"{self.platform}-{safe_filename(self.original_name)}"


class PoliteRateLimiter:
    """Garante intervalo mínimo entre requisições e pausas periódicas."""

    def __init__(
        self,
        delay_seconds: float,
        jitter_seconds: float,
        pause_every: int,
        pause_seconds: float,
    ) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self.jitter_seconds = max(0.0, jitter_seconds)
        self.pause_every = max(0, pause_every)
        self.pause_seconds = max(0.0, pause_seconds)
        self.request_count = 0
        self.last_request_at = 0.0

    def wait_before_request(self) -> None:
        minimum_wait = self.delay_seconds + random.uniform(0, self.jitter_seconds)
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < minimum_wait:
            time.sleep(minimum_wait - elapsed)

        if self.pause_every and self.request_count and self.request_count % self.pause_every == 0:
            logging.info(
                "Pausa preventiva de %.1f s após %d requisições.",
                self.pause_seconds,
                self.request_count,
            )
            time.sleep(self.pause_seconds)

        self.last_request_at = time.monotonic()
        self.request_count += 1


def safe_filename(name: str) -> str:
    """Mantém o nome remoto, removendo apenas caracteres inválidos no filesystem."""

    decoded = unquote(name).strip()
    decoded = decoded.replace("\n", "").replace("\r", "")
    decoded = UNSAFE_FILENAME_CHARS.sub("_", decoded)
    decoded = decoded.strip(" .")
    return decoded or "arquivo_sem_nome"


def filename_from_url(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    return safe_filename(path.rsplit("/", 1)[-1])


def filename_from_content_disposition(header: str | None) -> str | None:
    """Obtém filename/filename* quando o servidor informa nome explícito."""

    if not header:
        return None
    message = Message()
    message["content-disposition"] = header
    value = message.get_param("filename", header="content-disposition")
    if value:
        return safe_filename(str(value))
    return None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def response_headers(response: Any) -> dict[str, str]:
    return {key.lower(): value for key, value in response.headers.items()}


def retry_after_seconds(headers: Mapping[str, str], fallback: float) -> float:
    raw = headers.get("retry-after", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = fallback
    # Evita uma espera desproporcional causada por cabeçalho inesperado.
    return min(max(0.0, value), 300.0)


def http_open(
    url: str,
    limiter: PoliteRateLimiter,
    timeout_seconds: float,
    max_retries: int,
    headers: Mapping[str, str] | None = None,
) -> Any:
    """Abre URL pública com retry apenas para falhas transitórias."""

    request_headers = {
        "User-Agent": os.getenv("DADOS_ABERTOS_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept": "*/*",
        "Accept-Encoding": "identity",
    }
    if headers:
        request_headers.update(headers)

    for attempt in range(max_retries + 1):
        limiter.wait_before_request()
        request = Request(url, headers=request_headers, method="GET")
        try:
            return urlopen(request, timeout=timeout_seconds)
        except HTTPError as exc:
            if exc.code == 304:
                raise
            if exc.code not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
                raise
            fallback = min(60.0, 2.0 ** attempt)
            wait_seconds = retry_after_seconds(
                {key.lower(): value for key, value in exc.headers.items()}, fallback
            ) + random.uniform(0, 1.0)
            logging.warning(
                "HTTP %s em %s; nova tentativa em %.1f s (%d/%d).",
                exc.code,
                url,
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_seconds)
        except (URLError, TimeoutError, ConnectionError) as exc:
            if attempt >= max_retries:
                raise
            wait_seconds = min(60.0, 2.0 ** attempt) + random.uniform(0, 1.0)
            logging.warning(
                "Falha de rede em %s (%s); nova tentativa em %.1f s (%d/%d).",
                url,
                exc,
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Falha inesperada ao abrir {url}")


def fetch_json(
    url: str,
    limiter: PoliteRateLimiter,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    with http_open(url, limiter, timeout_seconds, max_retries, {"Accept": "application/json"}) as response:
        payload = response.read()
    parsed = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Resposta JSON inesperada em {url}")
    return parsed


def ckan_api_url(dataset_url: str) -> str:
    parts = urlsplit(dataset_url)
    return f"{parts.scheme}://{parts.netloc}/api/3/action/package_show?id={parts.path.rstrip('/').split('/')[-1]}"


def discover_ckan_resources(
    platform: str,
    dataset_url: str,
    limiter: PoliteRateLimiter,
    timeout_seconds: float,
    max_retries: int,
) -> list[RemoteFile]:
    api_url = ckan_api_url(dataset_url)
    payload = fetch_json(api_url, limiter, timeout_seconds, max_retries)
    if payload.get("success") is not True or not isinstance(payload.get("result"), dict):
        raise ValueError(f"API CKAN não retornou result válido: {api_url}")

    resources = payload["result"].get("resources", [])
    if not isinstance(resources, list):
        raise ValueError(f"Lista de recursos inválida: {api_url}")

    discovered: list[RemoteFile] = []
    seen_urls: set[str] = set()
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        url = str(resource.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        # O caminho da URL é a fonte mais confiável para manter o nome físico.
        original_name = filename_from_url(url)
        if not original_name or original_name == "arquivo_sem_nome":
            original_name = safe_filename(str(resource.get("name") or "arquivo_sem_nome"))
        discovered.append(RemoteFile(platform, url, original_name))

    return discovered


def same_host_and_descendant(url: str, root_url: str) -> bool:
    root = urlsplit(root_url)
    candidate = urlsplit(url)
    root_path = root.path if root.path.endswith("/") else root.path + "/"
    return (
        candidate.scheme in {"http", "https"}
        and candidate.netloc == root.netloc
        and candidate.path.startswith(root_path)
    )


def parse_directory_links(html: bytes, current_url: str) -> tuple[list[str], list[str]]:
    """Extrai links de diretórios Apache/nginx sem depender de parser externo."""

    # A listagem pública é HTML. O parser HTML da biblioteca padrão não existe;
    # esta expressão captura somente o atributo href de tags <a>, sem interpretar
    # conteúdo remoto como instrução.
    hrefs = re.findall(rb"<a\b[^>]*?href\s*=\s*([\"'])(.*?)\1", html, flags=re.I | re.S)
    directories: list[str] = []
    files: list[str] = []
    for _, raw_href in hrefs:
        href = raw_href.decode("utf-8", errors="replace").strip()
        if not href or href.startswith(("#", "?", "../")) or href in {".", "./", ".."}:
            continue
        absolute = urljoin(current_url, href)
        if absolute == current_url:
            continue
        if urlsplit(absolute).path.endswith("/"):
            directories.append(absolute)
        else:
            files.append(absolute)
    return directories, files


def discover_ibge_files(
    root_url: str,
    limiter: PoliteRateLimiter,
    timeout_seconds: float,
    max_retries: int,
) -> list[RemoteFile]:
    queue = [root_url]
    visited_dirs: set[str] = set()
    seen_files: set[str] = set()
    discovered: list[RemoteFile] = []

    while queue:
        current_url = queue.pop(0)
        normalized = current_url.split("#", 1)[0]
        if normalized in visited_dirs:
            continue
        visited_dirs.add(normalized)
        if not same_host_and_descendant(normalized, root_url):
            logging.warning("Ignorando link fora da raiz do IBGE: %s", normalized)
            continue

        with http_open(normalized, limiter, timeout_seconds, max_retries, {"Accept": "text/html"}) as response:
            html = response.read()

        directories, files = parse_directory_links(html, normalized)
        for directory in directories:
            if directory not in visited_dirs and same_host_and_descendant(directory, root_url):
                queue.append(directory)
        for file_url in files:
            if file_url in seen_files or not same_host_and_descendant(file_url, root_url):
                continue
            seen_files.add(file_url)
            discovered.append(RemoteFile("IBGE", file_url, filename_from_url(file_url)))

    return discovered


def resolve_target_names(resources: Iterable[RemoteFile]) -> list[RemoteFile]:
    """Garante nomes únicos; colisões recebem hash curto da URL de origem."""

    resolved: list[RemoteFile] = []
    used_names: dict[str, str] = {}
    for remote in resources:
        base_name = remote.target_name
        if base_name not in used_names:
            used_names[base_name] = remote.url
            resolved.append(remote)
            continue
        if used_names[base_name] == remote.url:
            continue

        path = Path(base_name)
        suffix = hashlib.sha256(remote.url.encode("utf-8")).hexdigest()[:10]
        candidate = f"{path.stem}__{suffix}{path.suffix}"
        counter = 2
        while candidate in used_names and used_names[candidate] != remote.url:
            candidate = f"{path.stem}__{suffix}_{counter}{path.suffix}"
            counter += 1
        used_names[candidate] = remote.url
        resolved.append(
            RemoteFile(
                platform=remote.platform,
                url=remote.url,
                original_name=remote.original_name,
                resolved_target_name=candidate,
            )
        )
    return resolved


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Manifesto inválido ou ilegível (%s); será recriado.", exc)
        return {}
    return data if isinstance(data, dict) else {}


def save_manifest(path: Path, manifest: Mapping[str, Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def platform_output_dir(output_root: str | Path, platform: str) -> Path:
    """Retorna o diretório da plataforma dentro da raiz do Volume."""

    try:
        subdirectory = PLATFORM_OUTPUT_DIRS[platform]
    except KeyError as exc:
        raise ValueError(f"Plataforma sem diretório configurado: {platform}") from exc
    return Path(output_root).expanduser() / subdirectory


def download_file(
    remote: RemoteFile,
    output_dir: Path,
    manifest: dict[str, dict[str, Any]],
    limiter: PoliteRateLimiter,
    timeout_seconds: float,
    max_retries: int,
    force: bool,
) -> str:
    target = output_dir / remote.target_name
    manifest_key = remote.url
    previous = manifest.get(manifest_key, {})
    conditional_headers: dict[str, str] = {}
    if not force and target.exists():
        if previous.get("etag"):
            conditional_headers["If-None-Match"] = str(previous["etag"])
        if previous.get("last_modified"):
            conditional_headers["If-Modified-Since"] = str(previous["last_modified"])

    try:
        response = http_open(
            remote.url,
            limiter,
            timeout_seconds,
            max_retries,
            conditional_headers,
        )
    except HTTPError as exc:
        if exc.code == 304 and target.exists():
            previous.update({"checked_at": iso_now(), "status": "not_modified"})
            manifest[manifest_key] = previous
            return "not_modified"
        raise

    headers = response_headers(response)
    server_name = filename_from_content_disposition(headers.get("content-disposition"))
    if server_name and remote.original_name == "arquivo_sem_nome":
        remote = RemoteFile(remote.platform, remote.url, server_name)
        target = output_dir / remote.target_name

    temporary = target.with_name(target.name + ".part")
    bytes_written = 0
    digest = hashlib.sha256()
    try:
        with response, temporary.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest[manifest_key] = {
        "platform": remote.platform,
        "source_url": remote.url,
        "target_name": target.name,
        "status": "downloaded",
        "downloaded_at": iso_now(),
        "size_bytes": bytes_written,
        "sha256": digest.hexdigest(),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "content_type": headers.get("content-type"),
        "content_length": headers.get("content-length"),
    }
    return "downloaded"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baixa arquivos públicos da ANEEL, ONS e IBGE para um Volume Databricks."
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("DADOS_ABERTOS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help=f"Diretório de saída (padrão: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=float(os.getenv("DADOS_ABERTOS_DELAY_SECONDS", "2.0")),
        help="Intervalo mínimo entre requisições, em segundos (padrão: 2).",
    )
    parser.add_argument(
        "--jitter-seconds",
        type=float,
        default=float(os.getenv("DADOS_ABERTOS_JITTER_SECONDS", "1.0")),
        help="Variação aleatória adicionada ao intervalo (padrão: 1).",
    )
    parser.add_argument(
        "--pause-every",
        type=int,
        default=int(os.getenv("DADOS_ABERTOS_PAUSE_EVERY", "25")),
        help="Faz pausa prolongada a cada N requisições (padrão: 25; 0 desativa).",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=float(os.getenv("DADOS_ABERTOS_PAUSE_SECONDS", "20")),
        help="Duração da pausa prolongada (padrão: 20).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("DADOS_ABERTOS_TIMEOUT_SECONDS", "120")),
        help="Timeout de cada requisição (padrão: 120).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("DADOS_ABERTOS_MAX_RETRIES", "4")),
        help="Número de novas tentativas para falhas transitórias (padrão: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora ETag/Last-Modified do manifesto e baixa novamente os arquivos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista os arquivos descobertos; não faz downloads.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limita a quantidade de arquivos para teste; 0 significa todos.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.getenv("DADOS_ABERTOS_LOG_LEVEL", "INFO"),
        help="Nível de log (padrão: INFO).",
    )
    return parser


def discover_all(
    limiter: PoliteRateLimiter,
    timeout_seconds: float,
    max_retries: int,
) -> list[RemoteFile]:
    sources: list[RemoteFile] = []
    sources.extend(
        discover_ckan_resources(
            "ANEEL", ANEEL_DATASET_URL, limiter, timeout_seconds, max_retries
        )
    )
    sources.extend(
        discover_ckan_resources(
            "ONS", ONS_DATASET_URL, limiter, timeout_seconds, max_retries
        )
    )
    sources.extend(discover_ibge_files(IBGE_ROOT_URL, limiter, timeout_seconds, max_retries))
    return sources


def discover_resources(
    limiter: PoliteRateLimiter,
    timeout_seconds: float,
    max_retries: int,
    max_files: int = 0,
) -> list[RemoteFile]:
    """Descobre, ordena e torna únicos os nomes dos arquivos."""

    resources = discover_all(limiter, timeout_seconds, max_retries)
    platform_order = {"ANEEL": 0, "ONS": 1, "IBGE": 2}
    resources.sort(
        key=lambda item: (platform_order.get(item.platform, 99), item.target_name, item.url)
    )
    resources = resolve_target_names(resources)
    if max_files:
        resources = resources[:max_files]
    return resources


def run_ingestion(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    delay_seconds: float = 2.0,
    jitter_seconds: float = 1.0,
    pause_every: int = 25,
    pause_seconds: float = 20.0,
    timeout_seconds: float = 120.0,
    max_retries: int = 4,
    force: bool = False,
    dry_run: bool = False,
    max_files: int = 0,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Executa a ingestão e retorna um resultado estruturado para notebook.

    O retorno contém `resources`, `summary`, `failures` e `manifest_paths`,
    facilitando a exibição com `display()` no Databricks.
    """

    if delay_seconds < 0 or jitter_seconds < 0 or pause_seconds < 0:
        raise ValueError("intervalos e pausas não podem ser negativos")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds deve ser maior que zero")
    if max_retries < 0 or max_files < 0:
        raise ValueError("max_retries e max_files não podem ser negativos")

    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"log_level inválido: {log_level}")
    logging.getLogger().setLevel(level)

    output_root = Path(output_dir).expanduser()
    platform_dirs = {
        platform: platform_output_dir(output_root, platform)
        for platform in PLATFORM_OUTPUT_DIRS
    }
    platform_manifest_paths = {
        platform: directory / ".ingest_dados_abertos_manifest.json"
        for platform, directory in platform_dirs.items()
    }
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        for directory in platform_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, Any]] = {}
    if not dry_run:
        for manifest_path in platform_manifest_paths.values():
            manifest.update(load_manifest(manifest_path))
    limiter = PoliteRateLimiter(
        delay_seconds=delay_seconds,
        jitter_seconds=jitter_seconds,
        pause_every=pause_every,
        pause_seconds=pause_seconds,
    )

    logging.info("Descobrindo recursos públicos nos catálogos da ANEEL, ONS e IBGE.")
    resources = discover_resources(
        limiter=limiter,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_files=max_files,
    )
    resource_rows = [
        {
            "platform": item.platform,
            "original_name": item.original_name,
            "target_name": item.target_name,
            "url": item.url,
        }
        for item in resources
    ]
    logging.info("Foram encontrados %d arquivos.", len(resources))
    for item in resources:
        logging.info("%s -> %s", item.platform, item.target_name)

    summary = {
        "discovered": len(resources),
        "downloaded": 0,
        "not_modified": 0,
        "failed": 0,
    }
    if dry_run:
        return {
            "resource_count": len(resources),
            "resources": resource_rows,
            "summary": summary,
            "failures": [],
            "manifest_paths": {
                platform: str(manifest_path)
                for platform, manifest_path in platform_manifest_paths.items()
            },
        }

    failures: list[dict[str, str]] = []
    for index, remote in enumerate(resources, start=1):
        logging.info("[%d/%d] Processando %s", index, len(resources), remote.target_name)
        try:
            status = download_file(
                remote=remote,
                output_dir=platform_dirs[remote.platform],
                manifest=manifest,
                limiter=limiter,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                force=force,
            )
            summary[status] = summary.get(status, 0) + 1
            save_manifest(
                platform_manifest_paths[remote.platform],
                {
                    key: value
                    for key, value in manifest.items()
                    if value.get("platform") == remote.platform
                },
            )
        except Exception as exc:  # continua para tentar os demais e reporta ao final
            summary["failed"] += 1
            failure = {
                "platform": remote.platform,
                "url": remote.url,
                "target_name": remote.target_name,
                "error": repr(exc),
            }
            failures.append(failure)
            logging.exception("Falha ao baixar %s: %s", remote.url, exc)
            manifest[remote.url] = {
                "platform": remote.platform,
                "source_url": remote.url,
                "target_name": remote.target_name,
                "status": "failed",
                "checked_at": iso_now(),
                "error": repr(exc),
            }
            save_manifest(
                platform_manifest_paths[remote.platform],
                {
                    key: value
                    for key, value in manifest.items()
                    if value.get("platform") == remote.platform
                },
            )

    logging.info(
        "Resumo: %d baixados, %d sem alteração, %d falhas; manifestos por plataforma: %s",
        summary.get("downloaded", 0),
        summary.get("not_modified", 0),
        summary.get("failed", 0),
        platform_manifest_paths,
    )
    return {
        "resource_count": len(resources),
        "resources": resource_rows,
        "summary": summary,
        "failures": failures,
        "manifest_paths": {
            platform: str(manifest_path)
            for platform, manifest_path in platform_manifest_paths.items()
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = run_ingestion(
        output_dir=args.output_dir,
        delay_seconds=args.delay_seconds,
        jitter_seconds=args.jitter_seconds,
        pause_every=args.pause_every,
        pause_seconds=args.pause_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        force=args.force,
        dry_run=args.dry_run,
        max_files=args.max_files,
        log_level=args.log_level,
    )
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
