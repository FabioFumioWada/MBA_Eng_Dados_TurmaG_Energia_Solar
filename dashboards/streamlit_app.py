
import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Previsão de Bandeiras | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# ENERGY INTELLIGENCE — DARK EXECUTIVE DASHBOARD
# Layout intentionally mirrors the approved visual reference.
# No real prediction probabilities are fabricated.
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #061522;
        --bg-2: #081d2e;
        --panel: #0a2133;
        --panel-2: #0c273b;
        --line: #16405d;
        --line-soft: #12334b;
        --white: #f4f8fc;
        --muted: #8fa8bb;
        --blue: #087cff;
        --blue-2: #16a0ff;
        --green: #00b86b;
        --yellow: #ffc400;
        --red: #ff3b4e;
        --orange: #ff7a00;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(8,124,255,.08), transparent 28%),
            radial-gradient(circle at 8% 35%, rgba(0,184,107,.035), transparent 25%),
            var(--bg);
        color: var(--white);
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        background: transparent;
    }

    /* Top brand */
    .brandbar {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:30px;
        padding: 4px 2px 20px 2px;
        border-bottom: 1px solid var(--line-soft);
    }

    .brand-left {
        display:flex;
        align-items:center;
        gap:14px;
    }

    .brand-mark {
        width:42px;
        height:42px;
        border:1px solid #1b6da4;
        border-radius:11px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:var(--blue-2);
        font-size:24px;
        font-weight:800;
        background:#071b2b;
    }

    .brand-name {
        font-size:18px;
        font-weight:800;
        letter-spacing:.4px;
        color:#f6fbff;
    }

    .brand-sub {
        margin-top:2px;
        color:#87a2b6;
        font-size:10px;
        letter-spacing:.2px;
    }

    .brand-meta {
        display:flex;
        align-items:center;
        gap:28px;
        color:#b5c7d5;
        font-size:11px;
        text-align:left;
    }

    .brand-meta-item {
        padding-left:20px;
        border-left:1px solid #1a3b53;
        line-height:1.45;
    }

    .brand-meta-dot {
        color:var(--blue);
        font-size:16px;
        vertical-align:middle;
        margin-right:7px;
    }

    /* Hero */
    .hero {
        position:relative;
        overflow:hidden;
        min-height:320px;
        margin-top:26px;
        padding:42px 38px 34px 38px;
        border:1px solid #145077;
        border-radius:18px;
        background:
            linear-gradient(90deg, rgba(6,21,34,.99) 0%, rgba(7,28,45,.96) 57%, rgba(5,30,48,.86) 100%);
        box-shadow:0 18px 50px rgba(0,0,0,.22);
    }

    .hero::after {
        content:"";
        position:absolute;
        inset:0;
        pointer-events:none;
        background:
            linear-gradient(90deg, transparent 62%, rgba(8,124,255,.04)),
            repeating-linear-gradient(120deg, transparent 0 80px, rgba(33,105,145,.025) 80px 81px);
    }

    .hero-kicker {
        position:relative;
        z-index:2;
        color:#1595ff;
        font-size:11px;
        font-weight:800;
        letter-spacing:2.2px;
        text-transform:uppercase;
        margin-bottom:12px;
    }

    .hero-title {
        position:relative;
        z-index:2;
        margin:0;
        max-width:760px;
        font-size:48px;
        line-height:1.02;
        font-weight:800;
        letter-spacing:-1.7px;
        color:#f5f8fb;
    }

    .hero-title .accent {
        color:#087cff;
    }

    .hero-sub {
        position:relative;
        z-index:2;
        max-width:690px;
        margin-top:15px;
        color:#bfd0dc;
        font-size:16px;
        line-height:1.5;
    }

    .hero-benefits {
        position:relative;
        z-index:2;
        display:flex;
        gap:28px;
        margin-top:28px;
        flex-wrap:wrap;
    }

    .benefit {
        display:flex;
        align-items:center;
        gap:10px;
        color:#dce8f0;
        font-size:13px;
        line-height:1.35;
        padding-right:26px;
        border-right:1px solid #15374e;
    }

    .benefit:last-child { border-right:0; }

    .benefit-icon {
        width:31px;
        height:31px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        background:#087cff;
        color:white;
        font-size:16px;
        flex:0 0 auto;
    }

    .hero-side {
        position:absolute;
        right:36px;
        top:100px;
        width:150px;
        z-index:2;
        border-left:2px solid #0d8dff;
        padding-left:16px;
        color:#9eb4c4;
        font-size:11px;
        line-height:1.75;
        letter-spacing:3px;
        text-transform:uppercase;
    }

    /* Business question */
    .question {
        display:flex;
        align-items:center;
        gap:24px;
        margin-top:20px;
        padding:23px 25px;
        border:1px solid #154563;
        border-radius:15px;
        background:#071c2c;
    }

    .question-bar {
        width:7px;
        min-height:64px;
        border-radius:7px;
        background:#087cff;
        flex:0 0 auto;
    }

    .eyebrow {
        color:#1195ff;
        font-size:9px;
        font-weight:800;
        letter-spacing:2px;
        text-transform:uppercase;
    }

    .question-text {
        margin-top:6px;
        color:#f1f6fa;
        font-size:17px;
        line-height:1.4;
        font-weight:700;
    }

    .question-action {
        margin-left:auto;
        min-width:130px;
        padding-left:24px;
        border-left:1px solid #1a415b;
        color:#e3edf4;
        font-size:12px;
        line-height:1.4;
    }

    .question-action span {
        color:#087cff;
        font-size:28px;
        vertical-align:middle;
        margin-right:8px;
    }

    /* Section */
    .section-head {
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        margin-top:27px;
        margin-bottom:13px;
    }

    .section-title {
        border-left:4px solid #087cff;
        padding-left:16px;
    }

    .section-title h2 {
        margin:0;
        color:#eef5fa;
        font-size:17px;
        font-weight:800;
        letter-spacing:.3px;
        text-transform:uppercase;
    }

    .section-title p {
        margin:5px 0 0;
        color:#829bad;
        font-size:11px;
    }

    .update {
        color:#93aabb;
        font-size:11px;
        text-align:right;
    }

    /* Cards */
    .cards {
        display:grid;
        grid-template-columns:repeat(4, 1fr);
        gap:14px;
    }

    .risk-card {
        position:relative;
        min-height:155px;
        overflow:hidden;
        border:1px solid #19425b;
        border-radius:14px;
        background:linear-gradient(180deg, #0b2639 0%, #081d2e 100%);
        box-shadow:0 10px 25px rgba(0,0,0,.14);
    }

    .risk-card::before {
        content:"";
        display:block;
        height:6px;
        background:var(--card-color);
    }

    .risk-inner {
        padding:19px 20px 17px;
    }

    .risk-label {
        color:#dce8ef;
        font-size:11px;
        font-weight:800;
        letter-spacing:1.3px;
        text-transform:uppercase;
    }

    .risk-value {
        margin-top:30px;
        color:#f6fbff;
        font-size:30px;
        line-height:1;
        font-weight:800;
    }

    .risk-note {
        margin-top:12px;
        color:#8da7b8;
        font-size:11px;
        line-height:1.4;
    }

    .risk-icon {
        position:absolute;
        right:18px;
        bottom:18px;
        color:#6b92ae;
        font-size:25px;
    }

    /* Impact */
    .impact {
        display:grid;
        grid-template-columns:1.05fr 2fr;
        gap:18px;
        margin-top:18px;
        padding:25px;
        border:1px solid #16405a;
        border-radius:15px;
        background:#071c2c;
    }

    .impact-intro h3 {
        margin:7px 0 10px;
        color:#f4f8fb;
        font-size:21px;
        line-height:1.18;
    }

    .impact-intro p {
        margin:0;
        color:#a8bccb;
        font-size:12px;
        line-height:1.65;
        max-width:330px;
    }

    .impact-button {
        display:inline-block;
        margin-top:17px;
        padding:9px 15px;
        border-radius:7px;
        background:#087cff;
        color:white;
        font-size:11px;
        font-weight:700;
    }

    .impact-grid {
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:12px;
    }

    .impact-card {
        min-height:150px;
        padding:18px;
        border:1px solid #19435c;
        border-radius:12px;
        background:#092337;
    }

    .impact-icon {
        color:#087cff;
        font-size:22px;
    }

    .impact-card h4 {
        margin:19px 0 8px;
        color:#f2f7fa;
        font-size:12px;
        letter-spacing:.5px;
    }

    .impact-card p {
        margin:0;
        color:#92aabb;
        font-size:11px;
        line-height:1.55;
    }

    /* Evidence cards */
    .evidence-grid {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:18px;
    }

    .chart-card {
        min-height:290px;
        padding:20px;
        border:1px solid #16405a;
        border-radius:15px;
        background:#071c2c;
    }

    .chart-title {
        color:#edf4f8;
        font-size:14px;
        font-weight:800;
    }

    .chart-sub {
        margin-top:4px;
        color:#819bad;
        font-size:10px;
    }

    .chart-placeholder {
        height:185px;
        margin-top:18px;
        border-radius:9px;
        border:1px dashed #214a63;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#5f7f95;
        font-size:11px;
        background:
            repeating-linear-gradient(0deg, transparent 0 35px, rgba(42,86,112,.14) 35px 36px);
    }

    .legend {
        display:flex;
        gap:16px;
        margin-top:12px;
        color:#93aabb;
        font-size:10px;
    }

    .dot {
        display:inline-block;
        width:8px;
        height:8px;
        border-radius:2px;
        margin-right:5px;
    }

    /* Closing */
    .closing {
        display:flex;
        align-items:center;
        gap:22px;
        margin-top:18px;
        padding:25px;
        border:1px solid #16405a;
        border-radius:15px;
        background:#071c2c;
    }

    .quote {
        color:#087cff;
        font-size:45px;
        font-weight:800;
        line-height:1;
    }

    .closing-text {
        color:#e7eef3;
        font-size:14px;
        line-height:1.55;
        font-weight:600;
    }

    .closing-text small {
        display:block;
        margin-top:5px;
        color:#718da2;
        font-size:10px;
        font-weight:400;
    }

    .closing-side {
        margin-left:auto;
        padding-left:25px;
        border-left:1px solid #1a415a;
        color:#7893a6;
        font-size:9px;
        line-height:1.8;
        letter-spacing:2px;
        text-transform:uppercase;
    }

    .footer {
        display:flex;
        justify-content:space-between;
        margin-top:30px;
        padding:0 3px;
        color:#5f7d91;
        font-size:9px;
        letter-spacing:1px;
        text-transform:uppercase;
    }

    /* Streamlit tabs */
    div[data-baseweb="tab-list"] {
        gap:4px;
        background:transparent;
        border-bottom:1px solid #153b54;
        margin-top:12px;
    }

    button[data-baseweb="tab"] {
        color:#7793a7 !important;
        background:transparent !important;
        border-radius:7px 7px 0 0 !important;
        padding:10px 15px !important;
        font-size:11px !important;
        font-weight:600 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color:#ffffff !important;
        background:#087cff !important;
    }

    div[data-baseweb="tab-highlight"] {
        display:none !important;
    }

    /* Native Streamlit text inside dark theme */
    .stMarkdown, .stText, label, p {
        color:inherit;
    }

    @media (max-width: 900px) {
        .hero-title { font-size:36px; }
        .hero-side { display:none; }
        .cards { grid-template-columns:repeat(2,1fr); }
        .impact, .evidence-grid { grid-template-columns:1fr; }
        .brand-meta { display:none; }
    }

    @media (max-width: 600px) {
        .cards { grid-template-columns:1fr; }
        .impact-grid { grid-template-columns:1fr; }
        .hero { padding:28px 22px; }
        .question-action { display:none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABRICKS CONNECTION
# ============================================================

from databricks import sql

CATALOG = "mba"
REFINED = f"{CATALOG}.refined"
TRUSTED = f"{CATALOG}.trusted"


def get_connection():
    """Open a read-only SQL connection using Streamlit Secrets."""
    return sql.connect(
        server_hostname=st.secrets["DATABRICKS_SERVER_HOSTNAME"],
        http_path=st.secrets["DATABRICKS_HTTP_PATH"],
        access_token=st.secrets["DATABRICKS_TOKEN"],
    )


@st.cache_data(ttl=300)
def query_df(query: str) -> pd.DataFrame:
    """Execute a SELECT query in the Databricks SQL Warehouse."""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_bandeiras():
    return query_df(f"""
        SELECT
            MesCompetencia,
            AnoMes,
            NivelBandeira,
            NomBandeiraAcionada,
            ValorAdicionalBandeira,
            EarPercentualNacional,
            EnaPercentualMltNacional,
            CmoMedioNacional,
            CargaTotalNacional,
            DatCarga
        FROM {REFINED}.painel_mensal_bandeira_hidrologia
        ORDER BY MesCompetencia
    """)


@st.cache_data(ttl=300)
def load_model_features():
    return query_df(f"""
        SELECT *
        FROM {REFINED}.modelo_previsao_bandeira
        WHERE ChuvaMedia_M2 IS NOT NULL
        ORDER BY MesCompetencia
    """)


@st.cache_data(ttl=300)
def load_clima():
    return query_df(f"""
        SELECT
            MesCompetencia,
            MesReferenciaClima,
            Mes,
            QtdEstacoesUsadas,
            PrecipitacaoMediaMm,
            PrecipitacaoAcumuladaMm,
            PrecipitacaoNormalMm,
            PrecipitacaoPctNormal,
            TemperaturaMediaC,
            UmidadeMediaPct,
            IsVermelha,
            DatCarga
        FROM {REFINED}.f_modelo_bandeira_clima
        ORDER BY MesCompetencia
    """)


@st.cache_data(ttl=300)
def load_training():
    return query_df(f"""
        SELECT *
        FROM {REFINED}.modelo_treino_m1
        ORDER BY MesCompetencia
    """)


def safe_date(df, column):
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column].astype(str), errors="coerce")
    return df


def flag_name(level):
    try:
        level = int(level)
    except Exception:
        return "—"
    return {
        0: "VERDE",
        1: "AMARELA",
        2: "VERMELHA P1",
        3: "VERMELHA P2",
        4: "ESCASSEZ HÍDRICA",
    }.get(level, f"NÍVEL {level}")


def pct(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.1f}%"


def html_cards(current_flag, current_date, model_status):
    return f"""
    <div class="cards">
      <div class="risk-card" style="--card-color:#00b86b;">
        <div class="risk-inner">
          <div class="risk-label">Bandeira atual</div>
          <div class="risk-value">{current_flag}</div>
          <div class="risk-note">{current_date}</div>
        </div>
        <div class="risk-icon">⚑</div>
      </div>

      <div class="risk-card" style="--card-color:#ff3b4e;">
        <div class="risk-inner">
          <div class="risk-label">Risco M+1</div>
          <div class="risk-value">—</div>
          <div class="risk-note">{model_status}</div>
        </div>
        <div class="risk-icon">▥</div>
      </div>

      <div class="risk-card" style="--card-color:#ffc400;">
        <div class="risk-inner">
          <div class="risk-label">Risco M+2</div>
          <div class="risk-value">—</div>
          <div class="risk-note">{model_status}</div>
        </div>
        <div class="risk-icon">▥</div>
      </div>

      <div class="risk-card" style="--card-color:#087cff;">
        <div class="risk-inner">
          <div class="risk-label">Risco M+3</div>
          <div class="risk-value">—</div>
          <div class="risk-note">{model_status}</div>
        </div>
        <div class="risk-icon">▥</div>
      </div>
    </div>
    """


def impact_section():
    return """
    <div class="impact">
      <div class="impact-intro">
        <div class="eyebrow">Por que isso importa?</div>
        <h3>A bandeira tarifária impacta diretamente o custo da energia.</h3>
        <p>
          Antecipar o risco permite transformar informação em planejamento,
          eficiência e maior resiliência para o setor elétrico.
        </p>
        <div class="impact-button">Entenda o problema&nbsp; →</div>
      </div>

      <div class="impact-grid">
        <div class="impact-card">
          <div class="impact-icon">●</div>
          <h4>CONSUMIDORES</h4>
          <p>Mais previsibilidade no planejamento financeiro.</p>
        </div>
        <div class="impact-card">
          <div class="impact-icon">▣</div>
          <h4>EMPRESAS</h4>
          <p>Maior eficiência operacional e redução de riscos.</p>
        </div>
        <div class="impact-card">
          <div class="impact-icon">◆</div>
          <h4>SISTEMA ELÉTRICO</h4>
          <p>Contribuição para um setor mais estável e sustentável.</p>
        </div>
      </div>
    </div>
    """


def closing_section():
    return """
    <div class="closing">
      <div class="quote">“</div>
      <div class="closing-text">
        O objetivo não é prever uma certeza.<br>
        É transformar dados disponíveis hoje em um sinal antecipado de risco para os próximos meses.
        <small>DADOS · ANÁLISE · PREVISÃO · DECISÃO</small>
      </div>
      <div class="closing-side">
        Dados<br>
        Análise<br>
        Previsão<br>
        Decisão
      </div>
    </div>

    <div class="footer">
      <div>Energy Intelligence · Mackenzie MBA · Engenharia de Dados</div>
      <div>Setor elétrico mais inteligente, decisões mais sustentáveis.</div>
    </div>
    """


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    """
    <div class="brandbar">
      <div class="brand-left">
        <div class="brand-mark">⌁</div>
        <div>
          <div class="brand-name">ENERGY INTELLIGENCE</div>
          <div class="brand-sub">Dados hoje. Decisões melhores amanhã.</div>
        </div>
      </div>

      <div class="brand-meta">
        <div class="brand-meta-item">
          <span class="brand-meta-dot">●</span>
          MACKENZIE MBA<br>
          Engenharia de Dados
        </div>
        <div class="brand-meta-item">
          Setor Elétrico Brasileiro<br>
          Bandeiras Tarifárias
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load Databricks data
# ------------------------------------------------------------

try:
    df_bandeiras = load_bandeiras()
    df_model = load_model_features()
    df_clima = load_clima()
    df_training = load_training()
    db_ok = True
    db_error = None
except Exception as exc:
    df_bandeiras = pd.DataFrame()
    df_model = pd.DataFrame()
    df_clima = pd.DataFrame()
    df_training = pd.DataFrame()
    db_ok = False
    db_error = str(exc)

if db_ok and not df_bandeiras.empty:
    df_bandeiras = safe_date(df_bandeiras, "MesCompetencia")
    latest = df_bandeiras.sort_values("MesCompetencia").iloc[-1]
    current_flag = flag_name(latest.get("NivelBandeira"))
    current_date = str(latest.get("MesCompetencia"))[:10]
    model_status = "Saída do modelo ainda não persistida em tabela"
else:
    current_flag = "—"
    current_date = "Databricks indisponível"
    model_status = "Conexão não disponível"


tabs = st.tabs(
    [
        "⌂  Visão Geral",
        "▣  Previsão",
        "⌁  Variáveis",
        "↗  Histórico",
        "●  Modelo",
        "▣  Metodologia",
    ]
)

# ------------------------------------------------------------
# TAB 1 — Visão Geral
# ------------------------------------------------------------

with tabs[0]:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Inteligência preditiva para o setor elétrico</div>
          <h1 class="hero-title">
            PREVISÃO DE<br>
            <span class="accent">BANDEIRAS TARIFÁRIAS</span>
          </h1>
          <div class="hero-sub">
            Transformando dados climáticos, hidrológicos e do sistema elétrico
            em sinais antecipados de risco.
          </div>

          <div class="hero-benefits">
            <div class="benefit">
              <span class="benefit-icon">⚡</span>
              Antecipação<br>de risco
            </div>
            <div class="benefit">
              <span class="benefit-icon">▥</span>
              Decisões<br>mais informadas
            </div>
            <div class="benefit">
              <span class="benefit-icon">◆</span>
              Contribuição para um<br>setor elétrico mais estável
            </div>
          </div>

          <div class="hero-side">
            ENERGIA<br>
            DADOS<br>
            PESSOAS<br>
            UM FUTURO<br>
            MAIS ESTÁVEL
          </div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div>
            <div class="eyebrow">Pergunta de negócio</div>
            <div class="question-text">
              Com as informações disponíveis hoje, conseguimos estimar o risco
              de bandeira vermelha em M+1, M+2 e M+3?
            </div>
          </div>
          <div class="question-action">
            <span>›</span> Explorar<br>previsões
          </div>
        </div>

        <div class="section-head">
          <div class="section-title">
            <h2>Painel executivo</h2>
            <p>Dados históricos e modelo conectados diretamente ao Databricks.</p>
          </div>
          <div class="update">Última atualização<br><strong>Databricks</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not db_ok:
        st.error("Não foi possível conectar ao Databricks SQL Warehouse.")
        st.caption("Verifique os três Secrets do aplicativo. Nenhum token é exibido pelo dashboard.")
    else:
        st.markdown(html_cards(current_flag, current_date, model_status), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Por que isso importa?</h2>
            <p>O valor do modelo está em antecipar um sinal de risco antes da decisão.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(impact_section(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Evidências no histórico</h2>
            <p>Os dados abaixo vêm diretamente das tabelas refinadas do Databricks.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_bandeiras.empty:
        chart = df_bandeiras[["MesCompetencia", "NivelBandeira"]].copy()
        chart["NivelBandeira"] = pd.to_numeric(chart["NivelBandeira"], errors="coerce")
        chart = chart.dropna().set_index("MesCompetencia")
        st.line_chart(chart["NivelBandeira"], use_container_width=True)
    else:
        st.warning("Histórico indisponível.")

    st.markdown(closing_section(), unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB 2 — Previsão
# ------------------------------------------------------------

with tabs[1]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Qual é o risco à frente?</h2>
            <p>A previsão transforma as variáveis disponíveis hoje em probabilidade de bandeira vermelha.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cards">
          <div class="risk-card" style="--card-color:#ff3b4e;">
            <div class="risk-inner">
              <div class="risk-label">M+1 · Próximo mês</div>
              <div class="risk-value">—</div>
              <div class="risk-note">Aguardando tabela de saída do modelo</div>
            </div>
            <div class="risk-icon">▥</div>
          </div>
          <div class="risk-card" style="--card-color:#ffc400;">
            <div class="risk-inner">
              <div class="risk-label">M+2 · Dois meses</div>
              <div class="risk-value">—</div>
              <div class="risk-note">Aguardando tabela de saída do modelo</div>
            </div>
            <div class="risk-icon">▥</div>
          </div>
          <div class="risk-card" style="--card-color:#087cff;">
            <div class="risk-inner">
              <div class="risk-label">M+3 · Três meses</div>
              <div class="risk-value">—</div>
              <div class="risk-note">Aguardando tabela de saída do modelo</div>
            </div>
            <div class="risk-icon">▥</div>
          </div>
          <div class="risk-card" style="--card-color:#087cff;">
            <div class="risk-inner">
              <div class="risk-label">Classe-alvo</div>
              <div class="risk-value" style="font-size:24px;">VERMELHA</div>
              <div class="risk-note">Classificação binária</div>
            </div>
            <div class="risk-icon">●</div>
          </div>
        </div>

        <div class="question" style="margin-top:18px;">
          <div class="question-bar"></div>
          <div>
            <div class="eyebrow">Interpretação</div>
            <div class="question-text" style="font-size:14px;">
              A probabilidade exibida nesta área deverá vir diretamente da saída
              persistida do modelo treinado no Databricks.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok:
        st.info(
            "Conexão com o Databricks está funcionando. A próxima integração é persistir "
            "o JSON final do notebook 07 em uma tabela de resultados para alimentar estes três cartões."
        )
    else:
        st.error("Sem conexão com o Databricks.")

# ------------------------------------------------------------
# TAB 3 — Variáveis
# ------------------------------------------------------------

with tabs[2]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Quais sinais entram na previsão?</h2>
            <p>As variáveis representam clima, hidrologia, sistema elétrico e histórico.</p>
          </div>
        </div>

        <div class="impact">
          <div class="impact-intro">
            <div class="eyebrow">Sinais do modelo</div>
            <h3>Dados disponíveis hoje para olhar o risco de amanhã.</h3>
            <p>
              O projeto combina variáveis observadas e transformações temporais
              para construir as features usadas na classificação.
            </p>
          </div>
          <div class="impact-grid">
            <div class="impact-card">
              <div class="impact-icon">≈</div>
              <h4>CLIMA</h4>
              <p>Precipitação, acumulado, normal climatológica, temperatura e umidade.</p>
            </div>
            <div class="impact-card">
              <div class="impact-icon">◆</div>
              <h4>HIDROLOGIA</h4>
              <p>EAR, ENA e suas variações temporais.</p>
            </div>
            <div class="impact-card">
              <div class="impact-icon">⚡</div>
              <h4>SISTEMA ELÉTRICO</h4>
              <p>CMO, carga e histórico de bandeira.</p>
            </div>
          </div>
        </div>

        <div class="section-head">
          <div class="section-title">
            <h2>Feature engineering</h2>
            <p>Features reais disponibilizadas pela camada refinada.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_model.empty:
        feature_cols = [
            c for c in [
                "EAR_M0", "EAR_M1", "EAR_M2",
                "ENA_M0", "ENA_M1", "ENA_M2",
                "CMO_M0", "CMO_M1",
                "Carga_M0", "Carga_M1",
                "ChuvaMedia_M0", "ChuvaMedia_M1", "ChuvaMedia_M2",
                "ChuvaAcumulada_M0", "ChuvaAcumulada_M1",
                "ChuvaPctNormal_M0", "ChuvaPctNormal_M1",
                "Temperatura_M0", "Umidade_M0",
                "ChuvaMedia_3M"
            ] if c in df_model.columns
        ]
        if feature_cols:
            st.dataframe(
                df_model[["MesCompetencia"] + feature_cols].tail(12),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("A tabela refinada foi encontrada, mas as colunas esperadas não estão disponíveis.")
    else:
        st.warning("Tabela de features indisponível.")

# ------------------------------------------------------------
# TAB 4 — Histórico
# ------------------------------------------------------------

with tabs[3]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Antes de prever o futuro, olhamos o passado</h2>
            <p>Evolução temporal das bandeiras e dos principais indicadores.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_bandeiras.empty:
        hist = df_bandeiras.copy()
        hist["MesCompetencia"] = pd.to_datetime(hist["MesCompetencia"], errors="coerce")
        hist["NivelBandeira"] = pd.to_numeric(hist["NivelBandeira"], errors="coerce")
        hist = hist.dropna(subset=["MesCompetencia", "NivelBandeira"])

        st.markdown("### Bandeiras ao longo do tempo")
        st.line_chart(
            hist.set_index("MesCompetencia")["NivelBandeira"],
            use_container_width=True,
        )

        st.markdown("### Indicadores hidrológicos e do sistema")
        cols = [
            c for c in [
                "EarPercentualNacional",
                "EnaPercentualMltNacional",
                "CmoMedioNacional",
                "CargaTotalNacional",
            ] if c in hist.columns
        ]
        if cols:
            numeric = hist[["MesCompetencia"] + cols].copy()
            for c in cols:
                numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
            st.line_chart(
                numeric.dropna(subset=["MesCompetencia"]).set_index("MesCompetencia")[cols],
                use_container_width=True,
            )

        st.dataframe(
            df_bandeiras.tail(12),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Histórico indisponível.")

# ------------------------------------------------------------
# TAB 5 — Modelo
# ------------------------------------------------------------

with tabs[4]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como transformamos sinais em previsão?</h2>
            <p>O modelo final utiliza clima, persistência da bandeira, EAR e marcos regulatórios.</p>
          </div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div style="width:100%;">
            <div class="eyebrow">Pipeline preditivo</div>
            <div class="question-text">
              Clima + Hidrologia + Histórico
              → Feature Engineering → Regressão Logística
              → Probabilidade M+1 / M+2 / M+3
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_training.empty:
        st.markdown("### Base de treinamento disponível")
        st.dataframe(
            df_training.tail(12),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"{len(df_training):,} linhas disponíveis na tabela refined.modelo_treino_m1."
        )
    else:
        st.warning("Tabela de treinamento indisponível.")

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Avaliação</h2>
            <p>Os resultados do backtest devem ser apresentados a partir da execução oficial do notebook de modelagem.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "A conexão do dashboard está separada da execução do modelo: o Streamlit lê dados "
        "do SQL Warehouse, enquanto o notebook 07 continua sendo a referência do modelo final."
    )

# ------------------------------------------------------------
# TAB 6 — Metodologia
# ------------------------------------------------------------

with tabs[5]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Da informação à decisão</h2>
            <p>Arquitetura e narrativa do projeto.</p>
          </div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div style="width:100%;">
            <div class="eyebrow">Arquitetura de dados</div>
            <div class="question-text">
              Fontes → Raw → Trusted → Refined → Feature Engineering →
              Modelo Preditivo → Probabilidade M+1 / M+2 / M+3 → Streamlit
            </div>
          </div>
        </div>

        <div class="impact" style="margin-top:18px;">
          <div class="impact-intro">
            <div class="eyebrow">Princípio</div>
            <h3>O objetivo não é prever uma certeza.</h3>
            <p>
              É transformar dados disponíveis hoje em um sinal antecipado
              de risco para apoiar decisões sobre os próximos meses.
            </p>
          </div>
          <div class="impact-grid">
            <div class="impact-card">
              <div class="impact-icon">01</div>
              <h4>FONTES</h4>
              <p>ANEEL, INMET e indicadores do sistema elétrico.</p>
            </div>
            <div class="impact-card">
              <div class="impact-icon">02</div>
              <h4>CAMADAS</h4>
              <p>Organização e tratamento em Raw, Trusted e Refined.</p>
            </div>
            <div class="impact-card">
              <div class="impact-icon">03</div>
              <h4>PRODUTO</h4>
              <p>Probabilidade de bandeira vermelha em três horizontes.</p>
            </div>
          </div>
        </div>

        <div class="section-head">
          <div class="section-title">
            <h2>Transparência</h2>
            <p>Os dados históricos são consultados no Databricks. As probabilidades futuras serão lidas da saída oficial do modelo.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(closing_section(), unsafe_allow_html=True)
