#!/usr/bin/env python3
"""API e cliente único para executar o Notebook 07 por ano e mês.

Modo servidor:
    python databricks_previsao.py --serve

Modo consulta única:
    python databricks_previsao.py --ano 2026 --mes 9

Configuração por variáveis de ambiente:
    DATABRICKS_HOST
    DATABRICKS_TOKEN                 (se ausente, será solicitado ocultamente)
    DATABRICKS_NOTEBOOK_PATH
    DATABRICKS_ENVIRONMENT_VERSION   (padrão: 5)
    DATABRICKS_POLL_SECONDS           (padrão: 5)
    DATABRICKS_MAX_WAIT_SECONDS       (padrão: 900)
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request


TERMINAL_STATES = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}


class DatabricksError(Exception):
    """Erro controlado da integração com o Databricks."""


class DatabricksTimeout(DatabricksError):
    """A execução não terminou dentro do prazo local configurado."""


@dataclass(frozen=True)
class Config:
    host: str
    token: str
    notebook_path: str
    environment_version: str = "5"
    poll_seconds: float = 5.0
    max_wait_seconds: float = 900.0


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise DatabricksError(f"Variável de ambiente ausente: {name}")
    return value


def load_env_file(path: str, *, required: bool = False) -> bool:
    """Carrega um .env simples; variáveis já existentes têm prioridade."""
    env_path = Path(path).expanduser()
    if not env_path.exists():
        if required:
            raise DatabricksError(f"Arquivo de configuração não encontrado: {env_path}")
        return False
    if not env_path.is_file():
        raise DatabricksError(f"O caminho de configuração não é um arquivo: {env_path}")

    valid_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatabricksError(f"Não foi possível ler {env_path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise DatabricksError(f"Linha inválida em {env_path}:{line_number}; use NOME=VALOR")

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not valid_name.fullmatch(name):
            raise DatabricksError(f"Nome de variável inválido em {env_path}:{line_number}: {name}")

        # Aceita valores entre aspas simples ou duplas. Comentários em linhas
        # separadas são aceitos; o valor não é alterado por interpolação.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)
    return True


def load_config(*, prompt_for_token: bool = True) -> Config:
    host = required_env("DATABRICKS_HOST").rstrip("/")
    notebook_path = required_env("DATABRICKS_NOTEBOOK_PATH")
    if not notebook_path.startswith("/"):
        raise DatabricksError("DATABRICKS_NOTEBOOK_PATH deve começar com '/'")

    token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    if not token and prompt_for_token:
        token = getpass.getpass("DATABRICKS_TOKEN (não será exibido): ").strip()
    if not token:
        raise DatabricksError(
            "Defina DATABRICKS_TOKEN ou execute o script em modo interativo para informá-lo."
        )

    try:
        poll_seconds = float(os.environ.get("DATABRICKS_POLL_SECONDS", "5"))
        max_wait_seconds = float(os.environ.get("DATABRICKS_MAX_WAIT_SECONDS", "900"))
    except ValueError as exc:
        raise DatabricksError("DATABRICKS_POLL_SECONDS e DATABRICKS_MAX_WAIT_SECONDS devem ser números") from exc
    if poll_seconds <= 0 or max_wait_seconds <= 0:
        raise DatabricksError("Os tempos de polling devem ser maiores que zero")

    return Config(
        host=host,
        token=token,
        notebook_path=notebook_path,
        environment_version=os.environ.get("DATABRICKS_ENVIRONMENT_VERSION", "5"),
        poll_seconds=poll_seconds,
        max_wait_seconds=max_wait_seconds,
    )


class DatabricksClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "Content-Type": "application/json",
            }
        )

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.host}{path}"
        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise DatabricksError(f"Falha de comunicação com o Databricks: {exc}") from exc

        if not response.ok:
            try:
                details = response.json()
                detail_text = json.dumps(details, ensure_ascii=False)
            except ValueError:
                detail_text = response.text[:1000]
            raise DatabricksError(f"Databricks HTTP {response.status_code}: {detail_text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise DatabricksError(f"Databricks não retornou JSON em {method} {path}") from exc
        if not isinstance(data, dict):
            raise DatabricksError(f"Databricks retornou JSON inesperado em {method} {path}")
        return data

    @staticmethod
    def state_of(run: dict[str, Any]) -> tuple[str, str, str]:
        status = run.get("status") or {}
        legacy = run.get("state") or {}
        termination = status.get("termination_details") or {}
        state = status.get("state") or legacy.get("life_cycle_state") or "UNKNOWN"
        result = termination.get("code") or legacy.get("result_state") or ""
        message = legacy.get("state_message") or termination.get("message") or ""
        return str(state), str(result), str(message)

    @staticmethod
    def child_run_id(run: dict[str, Any], task_key: str = "nb_07") -> int | str:
        tasks = run.get("tasks") or []
        for task in tasks:
            if task.get("task_key") == task_key and task.get("run_id") is not None:
                return task["run_id"]
        if len(tasks) == 1 and tasks[0].get("run_id") is not None:
            return tasks[0]["run_id"]
        raise DatabricksError(f"Não foi possível localizar o run_id da tarefa {task_key}")

    def run_prediction(
        self,
        year: int,
        month: int,
        *,
        idempotency_key: str | None = None,
        progress: bool = False,
    ) -> dict[str, Any]:
        period = f"{year:04d}-{month:02d}"
        if idempotency_key and len(idempotency_key) > 64:
            raise DatabricksError("Idempotency-Key deve ter no máximo 64 caracteres")
        safe_key = idempotency_key or f"api-{period}-{uuid.uuid4()}"

        payload: dict[str, Any] = {
            "run_name": f"Notebook 07 - consulta {period}",
            "idempotency_token": safe_key,
            "tasks": [
                {
                    "task_key": "nb_07",
                    "notebook_task": {
                        "notebook_path": self.config.notebook_path,
                        "source": "WORKSPACE",
                        "base_parameters": {
                            "ano": str(year),
                            "mes": f"{month:02d}",
                        },
                    },
                    "environment_key": "default",
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"environment_version": self.config.environment_version},
                }
            ],
        }

        submitted = self.request_json("POST", "/api/2.2/jobs/runs/submit", payload=payload)
        parent_run_id = submitted.get("run_id")
        if parent_run_id is None:
            raise DatabricksError("A submissão não retornou run_id")
        if progress:
            print(f"run_id={parent_run_id}", flush=True)

        started = time.monotonic()
        final_run: dict[str, Any] = {}
        state = "UNKNOWN"
        result_state = ""
        message = ""

        while True:
            final_run = self.request_json(
                "GET",
                "/api/2.2/jobs/runs/get",
                params={"run_id": parent_run_id},
            )
            state, result_state, message = self.state_of(final_run)
            if progress:
                suffix = f" result={result_state}" if result_state else ""
                print(f"state={state}{suffix}", flush=True)
            if state in TERMINAL_STATES:
                break
            if time.monotonic() - started >= self.config.max_wait_seconds:
                raise DatabricksTimeout(
                    f"run_id={parent_run_id} ainda está em {state}; consulte-o novamente depois"
                )
            time.sleep(self.config.poll_seconds)

        if result_state and result_state != "SUCCESS":
            raise DatabricksError(
                f"Notebook terminou sem sucesso: {message or f'state={state}, result={result_state}'}"
            )

        child_run_id = self.child_run_id(final_run)
        output = self.request_json(
            "GET",
            "/api/2.2/jobs/runs/get-output",
            params={"run_id": child_run_id},
        )
        notebook_output = output.get("notebook_output") or {}
        result_text = notebook_output.get("result")
        if result_text is None:
            error = output.get("error_trace") or output.get("error")
            raise DatabricksError(error or "Notebook não retornou notebook_output.result")

        try:
            result: Any = json.loads(result_text)
        except (TypeError, json.JSONDecodeError):
            result = {"texto": result_text}

        return {
            "status": "SUCCESS",
            "periodo_referencia": period,
            "run_id": parent_run_id,
            "task_run_id": child_run_id,
            "resultado": result,
            "truncated": bool(notebook_output.get("truncated", False)),
        }


def parse_period(body: Any) -> tuple[int, int]:
    if not isinstance(body, dict):
        raise ValueError("O corpo deve ser um JSON com ano e mes")
    if "ano" not in body or "mes" not in body:
        raise ValueError("Informe os campos ano e mes")
    try:
        year = int(body["ano"])
        month = int(body["mes"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ano e mes devem ser números inteiros") from exc
    if not 2000 <= year <= 2100:
        raise ValueError("ano deve estar entre 2000 e 2100")
    if not 1 <= month <= 12:
        raise ValueError("mes deve estar entre 1 e 12")
    return year, month


def create_app(client: DatabricksClient) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.post("/previsao")
    def previsao() -> Any:
        try:
            year, month = parse_period(request.get_json(silent=True))
            result = client.run_prediction(
                year,
                month,
                idempotency_key=request.headers.get("Idempotency-Key"),
                progress=False,
            )
            return jsonify(result), 200
        except ValueError as exc:
            return jsonify({"status": "ERROR", "erro": str(exc)}), 400
        except DatabricksTimeout as exc:
            return jsonify({"status": "TIMEOUT", "erro": str(exc)}), 504
        except DatabricksError as exc:
            return jsonify({"status": "ERROR", "erro": str(exc)}), 502

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o Notebook 07 por período ou inicia uma API local."
    )
    parser.add_argument("--serve", action="store_true", help="Inicia a API local em 127.0.0.1")
    parser.add_argument("--host", default="127.0.0.1", help="Host local da API (padrão: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Porta local da API (padrão: 8000)")
    parser.add_argument("--ano", type=int, help="Ano da consulta no modo de execução única")
    parser.add_argument("--mes", type=int, help="Mês da consulta no modo de execução única")
    parser.add_argument("--idempotency-key", help="Chave opcional, com no máximo 64 caracteres")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Arquivo de variáveis (padrão: .env na pasta atual)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_file_loaded = load_env_file(args.env_file, required=args.env_file != ".env")
    has_any_period = args.ano is not None or args.mes is not None
    has_both_period_values = args.ano is not None and args.mes is not None
    if args.serve and has_any_period:
        raise DatabricksError("Use --serve sem --ano/--mes")
    if not args.serve and not has_both_period_values:
        raise DatabricksError("Informe os dois argumentos: --ano e --mes, ou use --serve")

    config = load_config(prompt_for_token=True)
    client = DatabricksClient(config)

    if args.serve:
        app = create_app(client)
        if env_file_loaded:
            print(f"Configuração carregada de {Path(args.env_file)}", flush=True)
        print(f"API disponível em http://{args.host}:{args.port}", flush=True)
        print("Endpoint: POST /previsao com JSON {\"ano\": 2026, \"mes\": 9}", flush=True)
        app.run(host=args.host, port=args.port, debug=False)
        return 0

    result = client.run_prediction(
        args.ano,
        args.mes,
        idempotency_key=args.idempotency_key,
        progress=True,
    )
    if env_file_loaded:
        print(f"Configuração carregada de {Path(args.env_file)}", flush=True)
    print("resultado_json=", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Execução local interrompida; a execução remota pode continuar no Databricks.", file=sys.stderr)
        raise SystemExit(130)
    except (DatabricksError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1)
