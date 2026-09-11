#!/usr/bin/env python3
"""Portal local Streamlit para executar o Notebook 07 por mês e ano.

Execute a partir desta pasta:
    streamlit run portal_streamlit.py

Localmente, o arquivo .env fica em ../python/.env. No Streamlit Cloud,
as mesmas variáveis são lidas de st.secrets. O token nunca é exibido na tela.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PYTHON_DIR = APP_DIR.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from databricks_previsao import (  # noqa: E402
    DatabricksClient,
    DatabricksError,
    DatabricksTimeout,
    load_config,
    load_env_file,
    parse_period,
)


st.set_page_config(
    page_title="Previsão de Bandeira Tarifária",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        background: linear-gradient(135deg, #102a43 0%, #1f4e79 58%, #2d7d9a 100%);
        border-radius: 18px;
        padding: 2rem 2.25rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 14px 35px rgba(16, 42, 67, 0.18);
    }
    .hero h1 { margin: 0 0 .45rem 0; font-size: 2.15rem; letter-spacing: -.03em; }
    .hero p { margin: 0; color: #d9edf5; font-size: 1.03rem; }
    .section-label {
        color: #486581;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 1.4rem 0 .55rem 0;
    }
    .result-card {
        border: 1px solid #d9e2ec;
        border-radius: 14px;
        padding: 1.05rem 1.1rem;
        background: #ffffff;
        min-height: 185px;
        box-shadow: 0 5px 15px rgba(16, 42, 67, .06);
    }
    .result-card h3 { color: #102a43; margin: 0 0 .65rem 0; }
    .result-card .period { color: #627d98; font-size: .9rem; margin-bottom: .8rem; }
    .result-card .prediction { color: #102a43; font-size: 1.25rem; font-weight: 700; margin-bottom: .25rem; }
    .result-card .probability { color: #486581; font-size: .93rem; }
    .ok-pill { color: #176b4d; font-weight: 700; }
    .bad-pill { color: #b42318; font-weight: 700; }
    .muted { color: #627d98; font-size: .88rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_local_environment() -> tuple[bool, str]:
    env_path = PYTHON_DIR / ".env"
    try:
        loaded = load_env_file(str(env_path), required=False)
    except DatabricksError as exc:
        return False, str(exc)
    return loaded, str(env_path)


def load_streamlit_secrets() -> tuple[bool, str]:
    """Carrega secrets flat do Streamlit Cloud sem sobrescrever o ambiente."""
    try:
        secrets = dict(st.secrets)
    except Exception:
        return False, "Secrets do Streamlit não configurados"

    expected = (
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_NOTEBOOK_PATH",
        "DATABRICKS_ENVIRONMENT_VERSION",
        "DATABRICKS_POLL_SECONDS",
        "DATABRICKS_MAX_WAIT_SECONDS",
    )
    loaded = False
    for name in expected:
        value = secrets.get(name)
        if value is not None and str(value).strip():
            os.environ.setdefault(name, str(value))
            loaded = True
    if loaded:
        return True, "Secrets do Streamlit carregados"

    # Alguns ambientes de hospedagem expõem os secrets como variáveis de
    # ambiente. Reconhecemos esse caso para não exibir um diagnóstico falso.
    env_loaded = all(os.environ.get(name, "").strip() for name in expected[:3])
    if env_loaded:
        return True, "Variáveis seguras do ambiente carregadas"
    return False, "Secrets do Streamlit não configurados"


def percentage(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def display_prediction(horizon: str, prediction: dict[str, Any]) -> None:
    label = horizon.replace("t_plus_", "t+")
    class_name = prediction.get("classe_nome", "desconhecida")
    class_label = "Vermelha" if class_name == "vermelha" else "Não vermelha"
    real_value = prediction.get("valor_real")
    correct = prediction.get("acertou")
    if correct is True:
        validation = '<span class="ok-pill">✓ Acertou</span>'
    elif correct is False:
        validation = '<span class="bad-pill">✕ Não acertou</span>'
    else:
        validation = '<span class="muted">Ainda sem valor real</span>'

    st.markdown(
        f"""
        <div class="result-card">
            <h3>{label}</h3>
            <div class="period">Período alvo: <strong>{prediction.get('periodo_alvo', '—')}</strong></div>
            <div class="prediction">{class_label}</div>
            <div class="probability">Probabilidade vermelha: <strong>{percentage(prediction.get('probabilidade_vermelha'))}</strong></div>
            <div class="probability">Probabilidade não vermelha: <strong>{percentage(prediction.get('probabilidade_nao_vermelha'))}</strong></div>
            <div style="margin-top:.7rem">{validation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if real_value is not None:
        st.caption(f"Valor real codificado: {real_value} · Treino: {prediction.get('n_treino', '—')} observações")
    else:
        st.caption(f"Treino: {prediction.get('n_treino', '—')} observações")


def render_result(response: dict[str, Any]) -> None:
    result = response.get("resultado") or {}
    period = result.get("periodo_referencia", response.get("periodo_referencia", "—"))
    st.success(f"Execução concluída · período de referência {period}")

    predictions = result.get("previsoes") or {}
    if predictions:
        st.markdown('<div class="section-label">Previsões por horizonte</div>', unsafe_allow_html=True)
        columns = st.columns(3, gap="medium")
        for column, horizon in zip(columns, ("t_plus_1", "t_plus_2", "t_plus_3")):
            with column:
                prediction = predictions.get(horizon)
                if prediction:
                    display_prediction(horizon, prediction)
                else:
                    st.warning(f"Resultado ausente para {horizon}")

    st.markdown('<div class="section-label">Desempenho histórico do backtest</div>', unsafe_allow_html=True)
    backtest = result.get("backtest") or []
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

    with st.expander("Detalhes da execução"):
        st.write(f"**run_id:** `{response.get('run_id', '—')}`")
        st.write(f"**task_run_id:** `{response.get('task_run_id', '—')}`")
        st.write(f"**Saída truncada:** `{response.get('truncated', False)}`")

    with st.expander("JSON completo"):
        st.json(response)
        st.download_button(
            "Baixar resultado JSON",
            data=json.dumps(response, ensure_ascii=False, indent=2),
            file_name=f"previsao_{period.replace('-', '_')}.json",
            mime="application/json",
        )


loaded_env, env_message = load_local_environment()
loaded_secrets, secrets_message = load_streamlit_secrets()

st.markdown(
    """
    <div class="hero">
        <h1>Previsão de Bandeira Tarifária</h1>
        <p>Execute o modelo 07 no Databricks informando um mês de referência e consulte os horizontes t+1, t+2 e t+3.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Configuração local")
    if loaded_env or loaded_secrets:
        source = "arquivo .env" if loaded_env else secrets_message
        st.success(f"Configuração carregada: {source}")
    else:
        st.warning("Nenhuma configuração encontrada")
    st.caption(env_message if loaded_env else secrets_message)
    st.divider()
    st.markdown("**Como funciona**")
    st.caption("O portal envia os parâmetros ao notebook, acompanha a execução e exibe o JSON retornado.")
    st.caption("O token permanece no .env local e não é mostrado na tela.")

st.markdown('<div class="section-label">Período de referência</div>', unsafe_allow_html=True)
with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        year = st.number_input("Ano", min_value=2000, max_value=2100, value=2026, step=1)
    with right:
        month = st.number_input("Mês", min_value=1, max_value=12, value=8, step=1, format="%d")
    submitted = st.form_submit_button("Executar modelo", type="primary", use_container_width=True)

if submitted:
    try:
        year_value, month_value = parse_period({"ano": year, "mes": month})
        config = load_config(prompt_for_token=False)
        client = DatabricksClient(config)
        with st.spinner(f"Executando o notebook para {year_value:04d}-{month_value:02d}. Isso pode levar alguns minutos..."):
            response = client.run_prediction(year_value, month_value, progress=False)
        st.session_state["last_response"] = response
    except DatabricksTimeout as exc:
        st.error(str(exc))
        st.info("A execução pode continuar no Databricks. Use o run_id exibido no erro para consulta posterior.")
    except DatabricksError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)

if "last_response" in st.session_state:
    render_result(st.session_state["last_response"])
else:
    st.info("Informe o ano e o mês e clique em **Executar modelo** para começar.")
