#!/usr/bin/env python3
"""Portal Streamlit autônomo para o notebook 07 parametrizado.

Este arquivo reúne a interface, a leitura de configuração e o cliente da
Databricks Jobs API. Ele envia somente os cinco parâmetros expostos pelo
notebook 07_PP_regressao_logistica_clima_ear_regua_final_estruturado_parametrizado:

    chuva_acum
    chuva_pct_normal
    temperatura
    umidade
    chuva_media

Não há campos nem parâmetros de entrada para ano ou mês.

Execução local:
    streamlit run portal_previsoes.py

Dependências:
    pip install streamlit requests

O arquivo procura api/python/.env quando estiver na raiz do projeto ou na
pasta api/portal_previsoes. No Streamlit Community Cloud, configure os mesmos
nomes em Settings > Secrets.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import streamlit as st


PARAMETER_NAMES = (
    "chuva_acum",
    "chuva_pct_normal",
    "temperatura",
    "umidade",
    "chuva_media",
)
TERMINAL_STATES = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}


class DatabricksError(Exception):
    """Erro controlado da integração com o Databricks."""


class DatabricksTimeout(DatabricksError):
    """A execução não terminou no prazo configurado."""


@dataclass(frozen=True)
class Config:
    host: str
    token: str
    notebook_path: str
    environment_version: str = "5"
    poll_seconds: float = 5.0
    max_wait_seconds: float = 900.0


def load_env_file(path: Path, *, required: bool = False) -> bool:
    """Carrega um arquivo .env simples sem substituir variáveis existentes."""
    if not path.exists():
        if required:
            raise DatabricksError(f"Arquivo de configuração não encontrado: {path}")
        return False
    if not path.is_file():
        raise DatabricksError(f"O caminho de configuração não é um arquivo: {path}")

    valid_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatabricksError(f"Não foi possível ler {path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise DatabricksError(f"Linha inválida em {path}:{line_number}; use NOME=VALOR")

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not valid_name.fullmatch(name):
            raise DatabricksError(f"Nome de variável inválido em {path}:{line_number}: {name}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)
    return True


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise DatabricksError(f"Variável de ambiente ausente: {name}")
    return value


def load_config(notebook_env_name: str) -> Config:
    host = required_env("DATABRICKS_HOST").rstrip("/")
    token = required_env("DATABRICKS_TOKEN")
    notebook_path = required_env(notebook_env_name)
    if not notebook_path.startswith("/"):
        raise DatabricksError(f"{notebook_env_name} deve começar com '/'")

    try:
        poll_seconds = float(os.environ.get("DATABRICKS_POLL_SECONDS", "5"))
        max_wait_seconds = float(os.environ.get("DATABRICKS_MAX_WAIT_SECONDS", "900"))
    except ValueError as exc:
        raise DatabricksError("Os tempos de polling devem ser numéricos") from exc
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
    """Cliente mínimo para submeter e acompanhar um notebook parametrizado."""

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
        try:
            response = self.session.request(
                method,
                f"{self.config.host}{path}",
                params=params,
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise DatabricksError(f"Falha de comunicação com o Databricks: {exc}") from exc

        if not response.ok:
            try:
                details = json.dumps(response.json(), ensure_ascii=False)
            except ValueError:
                details = response.text[:1000]
            raise DatabricksError(f"Databricks HTTP {response.status_code}: {details}")

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

    def run_prediction(self, parameters: dict[str, Any]) -> dict[str, Any]:
        missing = [name for name in PARAMETER_NAMES if name not in parameters]
        unexpected = sorted(set(parameters) - set(PARAMETER_NAMES))
        if missing:
            raise DatabricksError(f"Parâmetros ausentes: {', '.join(missing)}")
        if unexpected:
            raise DatabricksError(f"Parâmetros não suportados: {', '.join(unexpected)}")
        if any(value is None for value in parameters.values()):
            raise DatabricksError("Os parâmetros não podem conter valores nulos")

        base_parameters = {name: str(parameters[name]) for name in PARAMETER_NAMES}
        payload: dict[str, Any] = {
            "run_name": "Notebook 07 parametrizado - previsão t+1",
            "idempotency_token": f"portal-previsoes-{uuid.uuid4()}",
            "tasks": [
                {
                    "task_key": "nb_07",
                    "notebook_task": {
                        "notebook_path": self.config.notebook_path,
                        "source": "WORKSPACE",
                        "base_parameters": base_parameters,
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

        started = time.monotonic()
        final_run: dict[str, Any] = {}
        while True:
            final_run = self.request_json(
                "GET",
                "/api/2.2/jobs/runs/get",
                params={"run_id": parent_run_id},
            )
            state, result_state, message = self.state_of(final_run)
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
            "run_id": parent_run_id,
            "task_run_id": child_run_id,
            "parametros": parameters,
            "resultado": result,
            "truncated": bool(notebook_output.get("truncated", False)),
        }


def load_configuration() -> tuple[bool, str]:
    app_dir = Path(__file__).resolve().parent
    candidates = (
        app_dir.parent / "python" / ".env",
        app_dir / "api" / "python" / ".env",
        Path.cwd() / "api" / "python" / ".env",
        Path.cwd().parent / "python" / ".env",
    )
    for path in candidates:
        if path.exists():
            try:
                load_env_file(path)
            except DatabricksError as exc:
                return False, str(exc)
            return True, f"Configuração carregada de {path}"

    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    names = (
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_NOTEBOOK_PATH",
        "DATABRICKS_PREVISOES_NOTEBOOK_PATH",
        "DATABRICKS_ENVIRONMENT_VERSION",
        "DATABRICKS_POLL_SECONDS",
        "DATABRICKS_MAX_WAIT_SECONDS",
    )
    loaded_from_secrets = False
    for name in names:
        value = secrets.get(name)
        if value is not None and str(value).strip():
            os.environ.setdefault(name, str(value))
            loaded_from_secrets = True

    has_connection = all(os.environ.get(name, "").strip() for name in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"))
    has_notebook = any(
        os.environ.get(name, "").strip()
        for name in ("DATABRICKS_PREVISOES_NOTEBOOK_PATH", "DATABRICKS_NOTEBOOK_PATH")
    )
    if loaded_from_secrets and has_connection and has_notebook:
        return True, "Secrets do Streamlit carregados"
    if has_connection and has_notebook:
        return True, "Variáveis seguras do ambiente carregadas"
    return False, "Configure api/python/.env ou os Secrets do Streamlit Community Cloud"


def percentage(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def notebook_period(response: dict[str, Any]) -> str:
    result = response.get("resultado") or {}
    return str(result.get("periodo_referencia") or "não informado pelo notebook")


def display_prediction(prediction: dict[str, Any], response: dict[str, Any]) -> None:
    class_name = prediction.get("classe_nome", "desconhecida")
    class_label = "Vermelha" if class_name == "vermelha" else "Não vermelha"
    real_value = prediction.get("valor_real")
    correct = prediction.get("acertou")
    if correct is True:
        validation = '<span class="ok-pill">✓ Acertou</span>'
    elif correct is False:
        validation = '<span class="bad-pill">✕ Não acertou</span>'
    else:
        validation = '<span class="muted">Ainda sem valor real disponível</span>'

    target_period = prediction.get("periodo_alvo") or "não informado"
    reference_period = notebook_period(response)
    st.markdown(
        f"""
        <div class="result-card">
            <h3>Previsão t+1</h3>
            <div class="period">Período retornado pelo notebook: <strong>{reference_period}</strong></div>
            <div class="period">Período previsto: <strong>{target_period}</strong></div>
            <div class="prediction">{class_label}</div>
            <div class="probability">Probabilidade vermelha: <strong>{percentage(prediction.get('probabilidade_vermelha'))}</strong></div>
            <div class="probability">Probabilidade não vermelha: <strong>{percentage(prediction.get('probabilidade_nao_vermelha'))}</strong></div>
            <div style="margin-top:.7rem">{validation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if real_value is not None:
        real_label = "Vermelha" if str(real_value) == "1" else "Não vermelha"
        st.caption(f"Valor real: {real_label} ({real_value}) · Treino: {prediction.get('n_treino', '—')} observações")
    else:
        st.caption(f"Valor real ainda não disponível · Treino: {prediction.get('n_treino', '—')} observações")


def render_result(response: dict[str, Any]) -> None:
    result = response.get("resultado") or {}
    period = notebook_period(response)
    st.success(f"Execução concluída · período retornado pelo notebook: {period}")

    prediction = (result.get("previsoes") or {}).get("t_plus_1")
    if prediction:
        st.markdown('<div class="section-label">Resultado da previsão</div>', unsafe_allow_html=True)
        display_prediction(prediction, response)
    else:
        st.warning("O notebook não retornou a previsão t+1.")

    backtest = result.get("backtest") or []
    st.markdown('<div class="section-label">Desempenho histórico do backtest</div>', unsafe_allow_html=True)
    if backtest:
        rows = [
            {
                "Horizonte": item.get("horizonte"),
                "Previsões": item.get("n"),
                "Corretas": item.get("corretas"),
                "Acurácia (%)": item.get("acuracia_pct"),
                "IC95% inferior (%)": item.get("ic95_lo"),
                "IC95% superior (%)": item.get("ic95_hi"),
            }
            for item in backtest
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("O notebook não retornou a seção de backtest.")

    with st.expander("Parâmetros enviados"):
        st.json(response.get("parametros") or {})
    with st.expander("Detalhes da execução"):
        st.write(f"**run_id:** `{response.get('run_id', '—')}`")
        st.write(f"**task_run_id:** `{response.get('task_run_id', '—')}`")
        st.write(f"**Saída truncada:** `{response.get('truncated', False)}`")
    with st.expander("JSON completo"):
        st.json(response)
        st.download_button(
            "Baixar resultado JSON",
            data=json.dumps(response, ensure_ascii=False, indent=2),
            file_name="previsao_parametrizada.json",
            mime="application/json",
        )


st.set_page_config(
    page_title="Portal de Previsões - Pesquisa",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero { background: linear-gradient(135deg, #102a43 0%, #1f4e79 58%, #2d7d9a 100%); border-radius: 18px; padding: 2rem 2.25rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 14px 35px rgba(16, 42, 67, 0.18); }
        .hero h1 { margin: 0 0 .45rem 0; font-size: 2.15rem; letter-spacing: -.03em; }
        .hero p { margin: 0; color: #d9edf5; font-size: 1.03rem; }
        .section-label { color: #486581; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 1.4rem 0 .55rem 0; }
        .result-card { border: 1px solid #d9e2ec; border-radius: 14px; padding: 1.05rem 1.1rem; background: #ffffff; min-height: 230px; box-shadow: 0 5px 15px rgba(16, 42, 67, .06); }
        .result-card h3 { color: #102a43; margin: 0 0 .65rem 0; }
        .result-card .period { color: #627d98; font-size: .9rem; margin-bottom: .8rem; }
        .result-card .prediction { color: #102a43; font-size: 1.25rem; font-weight: 700; margin-bottom: .25rem; }
        .result-card .probability { color: #486581; font-size: .93rem; }
        .ok-pill { color: #176b4d; font-weight: 700; }
        .bad-pill { color: #b42318; font-weight: 700; }
        .muted { color: #627d98; font-size: .88rem; }
        .parameter-note { color: #627d98; font-size: .88rem; margin-top: -.35rem; margin-bottom: .7rem; }
    </style>
    <div class="hero">
        <h1>Portal de Previsões - Pesquisa</h1>
        <p>Envie os parâmetros meteorológicos ao modelo 07 parametrizado no Databricks e consulte a previsão tarifária t+1.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

loaded, configuration_message = load_configuration()
with st.sidebar:
    st.markdown("### Configuração")
    if loaded:
        st.success("Configuração disponível")
    else:
        st.warning("Configuração incompleta")
    st.caption(configuration_message)
    st.divider()
    st.markdown("**Como funciona**")
    st.caption("O portal envia os cinco parâmetros do notebook, acompanha a execução remota e exibe o JSON retornado.")
    st.caption("O token fica no arquivo api/python/.env local ou em st.secrets no Streamlit Cloud.")
    st.caption("O notebook retorna o horizonte t1.")

st.markdown('<div class="section-label">Parâmetros da previsão</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="parameter-note">Informe as cinco variáveis de origem utilizadas pelo notebook 07 parametrizado.</div>',
    unsafe_allow_html=True,
)

with st.form("parameterized_prediction_form"):
    left, right = st.columns(2)
    with left:
        chuva_acum = st.number_input("Chuva acumulada (mm)", min_value=0.0, max_value=10000.0, value=100.0, step=0.1, format="%.2f", help="Precipitação acumulada observada no mês de referência.", key="chuva_acum")
        chuva_pct_normal = st.number_input("Percentual de chuva (%)", min_value=0.0, max_value=10000.0, value=100.0, step=0.1, format="%.2f", help="Percentual da precipitação em relação à normal climatológica.", key="chuva_pct_normal")
        temperatura = st.number_input("Temperatura média (°C)", min_value=-50.0, max_value=60.0, value=25.0, step=0.1, format="%.2f", help="Temperatura média do mês de referência.", key="temperatura")
    with right:
        umidade = st.number_input("Umidade média (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1, format="%.2f", help="Umidade média do mês de referência, entre 0 e 100%.", key="umidade")
        chuva_media = st.number_input("Chuva média (mm)", min_value=0.0, max_value=10000.0, value=100.0, step=0.1, format="%.2f", help="Precipitação média usada pelo modelo.", key="chuva_media")
        #st.caption("A bandeira de origem é fixada internamente em 0 pelo notebook.")
    submitted = st.form_submit_button("Executar modelo", type="primary", use_container_width=True)

if submitted:
    if not loaded:
        st.error("Não foi possível localizar a configuração do Databricks. Preencha api/python/.env localmente ou configure st.secrets no Streamlit Cloud.")
    else:
        parameters = {
            "chuva_acum": chuva_acum,
            "chuva_pct_normal": chuva_pct_normal,
            "temperatura": temperatura,
            "umidade": umidade,
            "chuva_media": chuva_media,
        }
        try:
            notebook_env_name = "DATABRICKS_PREVISOES_NOTEBOOK_PATH"
            if not os.environ.get(notebook_env_name, "").strip():
                notebook_env_name = "DATABRICKS_NOTEBOOK_PATH"
            client = DatabricksClient(load_config(notebook_env_name))
            with st.status("Executando o notebook parametrizado", expanded=True) as execution_status:
                st.write("Submetendo os parâmetros ao Databricks Jobs API…")
                response = client.run_prediction(parameters)
                execution_status.update(label="Execução concluída", state="complete", expanded=False)
            st.session_state["last_response"] = response
        except DatabricksTimeout as exc:
            st.error(str(exc))
            st.info("A execução pode continuar no Databricks. Consulte o run_id exibido no erro.")
        except DatabricksError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

if "last_response" in st.session_state:
    render_result(st.session_state["last_response"])
else:
    st.info("Informe os cinco parâmetros e clique em **Executar modelo** para iniciar uma previsão.")
