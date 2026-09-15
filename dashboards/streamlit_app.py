
import io
import base64
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    .brand-qr {
        display:flex;
        align-items:center;
        gap:12px;
        padding-left:18px;
        margin-left:2px;
        border-left:1px solid var(--line-soft);
    }

    .brand-qr img {
        width:104px;
        height:104px;
        display:block;
        background:#ffffff;
        padding:5px;
        border-radius:7px;
    }

    .brand-qr-label {
        color:#b5c7d5;
        font-size:9px;
        line-height:1.35;
        text-transform:uppercase;
        letter-spacing:.7px;
        white-space:nowrap;
        align-self:center;
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
    .benefit-icon.lightning {
        font-family: "Segoe UI Symbol", "Arial Unicode MS", sans-serif;
        font-size: 24px;
        line-height: 1;
        transform: rotate(0deg);
    }
    .benefit-icon.lightning-shape {
        width: 14px;
        height: 22px;
        display: inline-block;
        flex: 0 0 14px;
        background: #087cff;
        clip-path: polygon(58% 0%, 100% 0%, 67% 40%, 94% 40%, 35% 100%, 47% 54%, 18% 54%);
        transform: none;
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

    /* Jornada do projeto — timeline, ferramentas e cartões de decisão */
    .timeline {
        position:relative;
        margin-top:8px;
        padding-left:30px;
    }

    .timeline::before {
        content:"";
        position:absolute;
        left:9px;
        top:6px;
        bottom:6px;
        width:2px;
        background:var(--line);
    }

    .tl-item {
        position:relative;
        padding:0 0 24px 22px;
    }

    .tl-item:last-child { padding-bottom:0; }

    .tl-dot {
        position:absolute;
        left:-30px;
        top:2px;
        width:16px;
        height:16px;
        border-radius:50%;
        background:var(--panel);
        border:2px solid var(--blue);
    }

    .tl-dot.abandoned { border-color:var(--red); }
    .tl-dot.done { border-color:var(--green); }
    .tl-dot.warn { border-color:var(--orange); }

    .tl-tag {
        font-size:9px;
        font-weight:800;
        letter-spacing:1.6px;
        text-transform:uppercase;
        color:var(--blue-2);
    }

    .tl-tag.abandoned { color:var(--red); }
    .tl-tag.done { color:var(--green); }
    .tl-tag.warn { color:var(--orange); }

    .tl-title {
        color:var(--white);
        font-size:14px;
        font-weight:800;
        margin-top:5px;
        line-height:1.35;
    }

    .tl-text {
        color:var(--muted);
        font-size:12px;
        line-height:1.65;
        margin-top:5px;
        max-width:880px;
    }

    .tool-pill {
        display:inline-flex;
        align-items:center;
        padding:7px 14px;
        margin:4px 6px 4px 0;
        border-radius:20px;
        border:1px solid var(--line);
        background:var(--panel-2);
        color:#cfe0eb;
        font-size:11px;
        font-weight:700;
    }

    .impact-card.abandoned { border-left:3px solid var(--red); }
    .impact-card.done { border-left:3px solid var(--green); }

    @media (max-width: 900px) {
        .timeline { padding-left:24px; }
        .tl-dot { left:-24px; }
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
def load_model_source_data():
    """Load the same real sources used by the final notebook 07."""
    clima = query_df(f"""
        SELECT MesCompetencia, MesReferenciaClima, PrecipitacaoMediaMm,
               PrecipitacaoAcumuladaMm, PrecipitacaoPctNormal,
               TemperaturaMediaC, UmidadeMediaPct
        FROM {REFINED}.f_modelo_bandeira_clima
        ORDER BY MesCompetencia
    """)
    band = query_df(f"""
        SELECT MesCompetencia, IsVermelha
        FROM {TRUSTED}.f_bandeira
        ORDER BY MesCompetencia
    """)
    ear = query_df(f"""
        SELECT ear_data AS Data, id_subsistema,
               CAST(ear_verif_subsistema_percentual AS DOUBLE) AS Valor
        FROM {TRUSTED}.f_ear_subsistema
        WHERE id_subsistema = 'SE' AND ear_data IS NOT NULL
        ORDER BY ear_data
    """)
    return clima, band, ear


def periodo_yyyymm(value):
    """Normalize date/int/string values to pandas Period('YYYY-MM')."""
    if pd.isna(value):
        return pd.NaT
    text = str(value)[:10]
    if len(text) == 6 and text.isdigit():
        return pd.Period(text, freq="M")
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return parsed.to_period("M")


SIM_MARCOS = {
    "regime_gsf_pld": ("2018-12", None),
    "regime_faixas_2019": ("2019-06", None),
    "intervencao_pandemia": ("2020-05", "2020-11"),
    "intervencao_escassez": ("2021-09", "2022-04"),
}
SIM_FEATURES = [
    "mes_clima", "chuva_acum", "chuva_media", "chuva_pct_normal_ok",
    "temperatura", "umidade", "bandeira_origem", "ear_pct", *SIM_MARCOS.keys()
]


def marca_regime(meses, inicio, fim=None):
    v = meses >= pd.Period(inicio, "M")
    if fim is not None:
        v = v & (meses <= pd.Period(fim, "M"))
    return v.astype(int)


def montar_base_simulador(clima, band, ear):
    """Reproduce the feature construction of notebook 07 using real data."""
    c = clima.copy()
    b = band.copy()
    e = ear.copy()

    c["origem"] = c["MesReferenciaClima"].map(periodo_yyyymm)
    c["competencia"] = c["MesCompetencia"].map(periodo_yyyymm)
    b["periodo"] = b["MesCompetencia"].map(periodo_yyyymm)
    e["mes"] = pd.to_datetime(e["Data"], errors="coerce").dt.to_period("M")

    serie_band = (
        b.dropna(subset=["periodo"])
        .set_index("periodo")["IsVermelha"]
        .astype(float)
        .sort_index()
    )
    ear_se = e.dropna(subset=["mes"]).groupby("mes")["Valor"].mean()

    c["bandeira_origem"] = [
        float(serie_band.get(o - 1, np.nan)) for o in c["origem"]
    ]
    c = c.dropna(subset=["origem", "competencia", "bandeira_origem"]).copy()
    c["mes_clima"] = [p.month for p in c["origem"]]
    c["chuva_media"] = pd.to_numeric(c["PrecipitacaoMediaMm"], errors="coerce")
    c["chuva_acum"] = pd.to_numeric(c["PrecipitacaoAcumuladaMm"], errors="coerce")
    c["temperatura"] = pd.to_numeric(c["TemperaturaMediaC"], errors="coerce")
    c["umidade"] = pd.to_numeric(c["UmidadeMediaPct"], errors="coerce")
    c["ear_pct"] = [float(ear_se.get(o, np.nan)) for o in c["origem"]]

    # Same expanding rainfall-normal logic used by notebook 07:
    # only previous occurrences of the same calendar month.
    c = c.sort_values("origem").reset_index(drop=True)
    normals = []
    for i, row in c.iterrows():
        previous = c.loc[:i - 1]
        same_month = previous[
            previous["mes_clima"] == row["mes_clima"]
        ]["chuva_acum"].dropna()
        normals.append(float(same_month.mean()) if len(same_month) else np.nan)

    c["normal_expansiva"] = normals
    reported_pct = pd.to_numeric(c["PrecipitacaoPctNormal"], errors="coerce")
    c["chuva_pct_normal_ok"] = np.where(
        c["normal_expansiva"] > 0,
        c["chuva_acum"] / c["normal_expansiva"] * 100,
        reported_pct,
    )
    c["IsVermelha"] = [
        int(serie_band.get(p, np.nan)) if p in serie_band.index else np.nan
        for p in c["competencia"]
    ]
    c = c.dropna(subset=["IsVermelha"]).copy()
    c["IsVermelha"] = c["IsVermelha"].astype(int)
    return c, serie_band


def peso_amostra_sim(y, alvos):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    class_weights = {
        cls: n / (len(classes) * count)
        for cls, count in zip(classes, counts)
    }
    class_w = np.array([class_weights[v] for v in y])
    recent_w = np.where(
        pd.PeriodIndex(alvos) >= pd.Period("2024-04", "M"), 5.0, 1.0
    )
    return class_w * recent_w


def fit_model_sim(X, y, alvos):
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ])
    model.fit(X, y, clf__sample_weight=peso_amostra_sim(y, alvos))
    return model


def build_scenario_row(reference_month, scenario, horizon):
    """Build the full feature row expected by the notebook-07 recipe."""
    alvo = reference_month + horizon
    row = dict(scenario)
    for name in SIM_MARCOS:
        ini, fim = SIM_MARCOS[name]
        active = alvo >= pd.Period(ini, "M")
        if fim is not None:
            active = active and alvo <= pd.Period(fim, "M")
        row[name] = int(active)
    return row


def predict_scenario(sim_base, serie_band, reference_month, scenario, horizon):
    """Train the same notebook-07 recipe up to the reference month, then score a scenario."""
    eligible = []
    for _, row in sim_base.iterrows():
        origem = row["origem"]
        alvo = origem + horizon
        if alvo in serie_band.index and alvo <= reference_month:
            item = {c: row[c] for c in SIM_FEATURES if c not in SIM_MARCOS}
            item["alvo_mes"] = alvo
            item["y"] = int(serie_band.loc[alvo])
            eligible.append(item)

    train = pd.DataFrame(eligible)
    if len(train) < 36 or train["y"].nunique() < 2:
        raise ValueError(f"Base histórica insuficiente para M+{horizon}.")

    for name, (ini, fim) in SIM_MARCOS.items():
        train[name] = marca_regime(
            pd.PeriodIndex(train["alvo_mes"]), ini, fim
        )

    model = fit_model_sim(
        train[SIM_FEATURES],
        train["y"].values,
        train["alvo_mes"].values,
    )
    scenario_row = build_scenario_row(reference_month, scenario, horizon)
    x_scenario = pd.DataFrame([scenario_row])[SIM_FEATURES]
    probability = float(model.predict_proba(x_scenario)[0][1])
    return probability, reference_month + horizon, len(train)


GITHUB_PROJECT_URL = "https://github.com/LaianeNR/MBA_Eng_Dados_TurmaG_Energia_Solar/tree/desenv"


def get_qr_data_uri():
    """Generate the project QR as an embeddable PNG data URI."""
    try:
        import qrcode

        qr = qrcode.QRCode(version=None, box_size=7, border=2)
        qr.add_data(GITHUB_PROJECT_URL)
        qr.make(fit=True)
        img = qr.make_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def render_qr():
    """QR Code for the project GitHub link requested for the presentation."""
    url = GITHUB_PROJECT_URL
    if not url:
        st.info("O QR Code aparecerá quando o app estiver publicado.")
        return

    try:
        import qrcode

        qr = qrcode.QRCode(version=None, box_size=8, border=3)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        st.image(buf.getvalue(), width=180)
        st.caption(
            f"Aponte a câmera do celular para abrir o dashboard.\n{url}"
        )
    except Exception as exc:
        st.warning(f"Não foi possível gerar o QR Code: {exc}")


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


# ------------------------------------------------------------
# STORYTELLING DASHBOARD
# ------------------------------------------------------------

PROJECT_REFERENCE = pd.Period("2026-08", "M")


def period_label(p):
    if p is None or pd.isna(p):
        return "—"
    return pd.Period(p, "M").strftime("%m/%Y")


def month_label_pt(p):
    names = {
        1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril",
        5: "maio", 6: "junho", 7: "julho", 8: "agosto",
        9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro",
    }
    try:
        p = pd.Period(p, "M")
        return f"{names[p.month]} de {p.year}"
    except Exception:
        return "—"


def month_index(df, primary="MesCompetencia", fallback="AnoMes"):
    """Usa MesCompetencia e, quando vier nulo, cai para AnoMes."""
    out = df.copy()
    primary_dt = (
        pd.to_datetime(out[primary].astype("string"), errors="coerce")
        if primary in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    fallback_dt = (
        pd.to_datetime(out[fallback].astype("string"), errors="coerce")
        if fallback in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    out["_mes_dashboard"] = primary_dt.fillna(fallback_dt)
    return out


def get_reference_row(sim_base, ref):
    rows = sim_base[sim_base["origem"] == ref].sort_values("competencia")
    if rows.empty:
        raise ValueError(f"Não há dados climáticos para a referência {period_label(ref)}.")
    return rows.iloc[0]


def calculate_real_forecast(sim_base, sim_series, ref):
    """Executive forecast using the same feature recipe as notebook 07."""
    r = get_reference_row(sim_base, ref)
    scenario = {
        "mes_clima": int(ref.month),
        "chuva_acum": float(r["chuva_acum"]) if pd.notna(r["chuva_acum"]) else np.nan,
        "chuva_media": float(r["chuva_media"]) if pd.notna(r["chuva_media"]) else np.nan,
        "chuva_pct_normal_ok": float(r["chuva_pct_normal_ok"]) if pd.notna(r["chuva_pct_normal_ok"]) else np.nan,
        "temperatura": float(r["temperatura"]) if pd.notna(r["temperatura"]) else np.nan,
        "umidade": float(r["umidade"]) if pd.notna(r["umidade"]) else np.nan,
        "bandeira_origem": float(r["bandeira_origem"]) if pd.notna(r["bandeira_origem"]) else 0.0,
        "ear_pct": float(r["ear_pct"]) if pd.notna(r["ear_pct"]) else np.nan,
    }
    out = []
    for h in (1, 2, 3):
        try:
            p, target, n_train = predict_scenario(sim_base, sim_series, ref, scenario, h)
            out.append({"h": h, "prob": p, "target": target, "n_train": n_train, "ok": True})
        except Exception as exc:
            out.append({"h": h, "prob": None, "target": ref + h, "n_train": None, "ok": False, "error": str(exc)})
    return scenario, out


def render_forecast_cards(results, title="Risco estimado de bandeira vermelha"):
    cols = st.columns(3)
    for col, item in zip(cols, results):
        with col:
            target = period_label(item["target"])
            if item.get("prob") is None:
                st.metric(f"M+{item['h']} · {target}", "—")
                if item.get("error"):
                    st.caption("Resultado indisponível para este horizonte.")
            else:
                st.metric(f"M+{item['h']} · {target}", f"{item['prob']:.1%}")
                st.caption(title)
                st.progress(min(max(float(item["prob"]), 0.0), 1.0))
                st.caption(f"Base de treino até {period_label(PROJECT_REFERENCE)} · {item['n_train']} observações")


def render_kpi_card(label, value, note, accent="#087cff"):
    return f"""
    <div class="risk-card" style="--card-color:{accent};">
      <div class="risk-inner">
        <div class="risk-label">{label}</div>
        <div class="risk-value">{value}</div>
        <div class="risk-note">{note}</div>
      </div>
    </div>
    """


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
qr_uri = get_qr_data_uri()
qr_html = (
    f'<div class="brand-qr">'
    f'<img src="{qr_uri}" alt="QR Code do projeto">'
    f'<div class="brand-qr-label">Acesse<br>o projeto</div>'
    f'</div>'
    if qr_uri else ""
)

st.markdown(
    f"""
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
        {qr_html}
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
    df_sim_clima, df_sim_band, df_sim_ear = load_model_source_data()
    db_ok = True
    db_error = None
except Exception as exc:
    df_bandeiras = pd.DataFrame()
    df_model = pd.DataFrame()
    df_clima = pd.DataFrame()
    df_training = pd.DataFrame()
    df_sim_clima = pd.DataFrame()
    df_sim_band = pd.DataFrame()
    df_sim_ear = pd.DataFrame()
    db_ok = False
    db_error = str(exc)

if db_ok and not df_bandeiras.empty:
    df_bandeiras = month_index(df_bandeiras)
    latest = df_bandeiras.dropna(subset=["_mes_dashboard"]).sort_values("_mes_dashboard").iloc[-1]
    current_flag = flag_name(latest.get("NivelBandeira"))
    current_date = period_label(latest.get("_mes_dashboard"))
else:
    current_flag = "—"
    current_date = "—"

# The presentation reference is explicitly August/2026.
reference_month = PROJECT_REFERENCE
reference_available = False
sim_base = pd.DataFrame()
sim_series = pd.Series(dtype=float)
forecast_results = []
forecast_scenario = {}

if db_ok and not df_sim_clima.empty and not df_sim_band.empty and not df_sim_ear.empty:
    try:
        sim_base, sim_series = montar_base_simulador(df_sim_clima, df_sim_band, df_sim_ear)
        reference_available = reference_month in set(sim_base["origem"].dropna().unique())
        if reference_available:
            forecast_scenario, forecast_results = calculate_real_forecast(sim_base, sim_series, reference_month)
    except Exception as exc:
        reference_available = False
        db_error = str(exc)

# ------------------------------------------------------------
# Tabs — ordered as the presentation story
# ------------------------------------------------------------
tabs = st.tabs([
    "01  Contexto",
    "02  Metodologia",
    "03  Jornada completa",
    "04  Histórico",
    "05  Sinais",
    "06  Modelo",
    "07  Previsão",
])

def flag_history_chart(df):
    """Evolução histórica com a linha segmentada pela cor da bandeira oficial."""
    hist = month_index(df)
    hist["NivelBandeira"] = pd.to_numeric(hist["NivelBandeira"], errors="coerce")
    hist = hist.dropna(subset=["_mes_dashboard", "NivelBandeira"]).sort_values("_mes_dashboard")
    if hist.empty:
        st.warning("Histórico sem dados válidos para visualização.")
        return

    colors = {0: "#00b86b", 1: "#ffc400", 2: "#ff3b4e", 3: "#ff3b4e", 4: "#ff7a00"}
    names = {0: "Verde", 1: "Amarela", 2: "Vermelha P1", 3: "Vermelha P2", 4: "Escassez Hídrica"}

    fig = go.Figure()
    x = hist["_mes_dashboard"].tolist()
    y = hist["NivelBandeira"].astype(float).tolist()

    # Segmentos coloridos: cada trecho assume a cor da bandeira observada no mês de origem.
    for i in range(len(x) - 1):
        level = int(round(y[i]))
        fig.add_trace(go.Scatter(
            x=[x[i], x[i + 1]],
            y=[y[i], y[i + 1]],
            mode="lines",
            line=dict(color=colors.get(level, "#087cff"), width=3),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Pontos coloridos e legenda oficial.
    for level in sorted(colors):
        mask = hist["NivelBandeira"].round().astype(int) == level
        if mask.any():
            fig.add_trace(go.Scatter(
                x=hist.loc[mask, "_mes_dashboard"],
                y=hist.loc[mask, "NivelBandeira"],
                mode="markers",
                name=names[level],
                marker=dict(color=colors[level], size=7, line=dict(width=1, color="#061522")),
                hovertemplate="<b>%{x|%m/%Y}</b><br>" + names[level] + "<extra></extra>",
            ))

    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="#07131f",
        plot_bgcolor="#07131f",
        font=dict(color="#cfe0eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, color="#8fa8bb"),
        yaxis=dict(
            title="Nível da bandeira",
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["Verde", "Amarela", "Vermelha P1", "Vermelha P2", "Escassez"],
            gridcolor="#183042",
            zeroline=False,
            color="#8fa8bb",
            range=[-0.2, 4.35],
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------
# TAB 1 — CONTEXTO
# ------------------------------------------------------------
with tabs[0]:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Inteligência preditiva para o setor elétrico</div>
          <h1 class="hero-title">PREVISÃO DE<br><span class="accent">BANDEIRAS TARIFÁRIAS</span></h1>
          <div class="hero-sub">
            Transformando sinais climáticos, hidrológicos e históricos em uma estimativa
            antecipada de risco para os próximos meses.
          </div>
          <div class="hero-benefits">
            <div class="benefit"><span class="benefit-icon lightning-shape" aria-hidden="true"></span>Antecipação<br>de risco</div>
            <div class="benefit"><span class="benefit-icon">▥</span>Decisões<br>mais informadas</div>
            <div class="benefit"><span class="benefit-icon">◆</span>Leitura integrada<br>do sistema elétrico</div>
          </div>
          <div class="hero-side">ENERGIA<br>DADOS<br>RISCO<br>DECISÃO</div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div>
            <div class="eyebrow">Pergunta de negócio</div>
            <div class="question-text">
              Com as informações disponíveis ao fim de agosto de 2026, conseguimos estimar
              o risco de bandeira vermelha em setembro, outubro e novembro?
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Por que esse problema importa?</h2>
            <p>A bandeira tarifária transforma condições do sistema elétrico em impacto econômico.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(impact_section(), unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-title">
            <h2>Ponto de partida</h2>
            <p>O projeto começa observando o comportamento que queremos antecipar.</p>
          </div>
          <div class="update">Referência do modelo<br><strong>{period_label(reference_month)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(4, gap="small")
    kpi_data = [
        ("Última bandeira observada", current_flag, f"Competência {current_date}", "#00b86b"),
        ("Horizonte 1", period_label(reference_month + 1), "Primeiro mês à frente", "#ff3b4e"),
        ("Horizonte 2", period_label(reference_month + 2), "Segundo mês à frente", "#ffc400"),
        ("Horizonte 3", period_label(reference_month + 3), "Terceiro mês à frente", "#087cff"),
    ]
    for col, (label, value, note, accent) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(render_kpi_card(label, value, note, accent), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>O que vamos investigar?</h2>
            <p>Antes de chegar à previsão, entendemos o comportamento histórico e os sinais disponíveis.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">01</div><h4>COMPORTAMENTO</h4><p>Como as bandeiras variaram ao longo do tempo e quais períodos se destacam?</p></div>
          <div class="impact-card"><div class="impact-icon">02</div><h4>SINAIS</h4><p>Quais informações estavam disponíveis antes de cada mudança de bandeira?</p></div>
          <div class="impact-card"><div class="impact-icon">03</div><h4>PREVISÃO</h4><p>Quanto esses sinais conseguem antecipar a probabilidade de bandeira vermelha?</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# TAB 4 — HISTÓRICO
# ------------------------------------------------------------
with tabs[3]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>O passado mostra o comportamento do risco</h2>
            <p>Antes de prever o futuro, observamos frequência, intensidade e mudanças de regime.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if db_ok and not df_bandeiras.empty:
        hist = month_index(df_bandeiras)
        hist["NivelBandeira"] = pd.to_numeric(hist["NivelBandeira"], errors="coerce")
        hist = hist.dropna(subset=["_mes_dashboard", "NivelBandeira"]).sort_values("_mes_dashboard")

        st.markdown("### Evolução das bandeiras")
        st.caption("A cor acompanha o nível oficial observado em cada mês.")
        flag_history_chart(df_bandeiras)

        dist = hist.copy()
        dist["Bandeira"] = dist["NivelBandeira"].map(flag_name)
        dist = dist["Bandeira"].value_counts().rename_axis("Bandeira").reset_index(name="Meses")
        dist["Percentual"] = dist["Meses"] / dist["Meses"].sum() * 100
        st.markdown("### Frequência das bandeiras")
        st.dataframe(dist, use_container_width=True, hide_index=True,
                     column_config={"Percentual": st.column_config.NumberColumn("Percentual", format="%.1f%%")})

        st.markdown("### Um período que mudou o comportamento da série")
        st.info("Entre maio e novembro de 2020, a série atravessou um período excepcional associado à pandemia. No protocolo de modelagem, esse intervalo é tratado como uma marca de regime para evitar que um comportamento fora do padrão seja confundido com a dinâmica estrutural.")

        st.markdown("### Indicadores do sistema elétrico")
        cols = [c for c in ["EarPercentualNacional", "EnaPercentualMltNacional", "CmoMedioNacional", "CargaTotalNacional"] if c in hist.columns]
        if cols:
            numeric = hist[["_mes_dashboard"] + cols].copy()
            for c in cols:
                numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
            st.line_chart(numeric.dropna(subset=["_mes_dashboard"]).set_index("_mes_dashboard")[cols], use_container_width=True)

        st.markdown("### Últimos registros observados")
        latest_view = df_bandeiras.drop(columns=["_mes_dashboard"], errors="ignore").tail(12)
        st.dataframe(latest_view, use_container_width=True, hide_index=True)
    else:
        st.warning("Histórico indisponível.")

# ------------------------------------------------------------
# TAB 5 — SINAIS
# ------------------------------------------------------------
with tabs[4]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Quais sinais estavam disponíveis antes da previsão?</h2>
            <p>O modelo transforma observações do mês de referência em atributos comparáveis.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">≈</div><h4>CLIMA</h4><p>Chuva média, chuva acumulada, percentual da normal, temperatura e umidade.</p></div>
          <div class="impact-card"><div class="impact-icon">◆</div><h4>HIDROLOGIA</h4><p>EAR do subsistema Sudeste, usado como sinal do estado dos reservatórios.</p></div>
          <div class="impact-card"><div class="impact-icon">⚑</div><h4>PERSISTÊNCIA</h4><p>Indicador de bandeira vermelha do mês anterior, capturando a memória de curto prazo da série.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_model.empty:
        feature_cols = [c for c in [
            "EAR_M0", "EAR_M1", "EAR_M2", "ENA_M0", "ENA_M1", "ENA_M2",
            "CMO_M0", "CMO_M1", "Carga_M0", "Carga_M1",
            "ChuvaMedia_M0", "ChuvaMedia_M1", "ChuvaMedia_M2",
            "ChuvaAcumulada_M0", "ChuvaAcumulada_M1",
            "ChuvaPctNormal_M0", "ChuvaPctNormal_M1",
            "Temperatura_M0", "Umidade_M0", "ChuvaMedia_3M"
        ] if c in df_model.columns]
        if feature_cols:
            st.markdown("### Base refinada usada na modelagem")
            st.dataframe(df_model[["MesCompetencia"] + feature_cols].tail(12), use_container_width=True, hide_index=True)

    if reference_available:
        r = get_reference_row(sim_base, reference_month)
        st.markdown(f"### Retrato de {month_label_pt(reference_month)}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Chuva acumulada", f"{float(r['chuva_acum']):.1f} mm" if pd.notna(r['chuva_acum']) else "—")
        k2.metric("Chuva vs. normal", f"{float(r['chuva_pct_normal_ok']):.1f}%" if pd.notna(r['chuva_pct_normal_ok']) else "—")
        k3.metric("EAR SE", f"{float(r['ear_pct']):.1f}%" if pd.notna(r['ear_pct']) else "—")
        k4.metric("Bandeira anterior", "Vermelha" if int(r['bandeira_origem']) == 1 else "Não vermelha")

        st.caption("Esses valores são os sinais observados no mês de referência. O modelo não utiliza informação do alvo futuro para construir a previsão.")

# ------------------------------------------------------------
# TAB 6 — MODELO
# ------------------------------------------------------------
with tabs[5]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como os sinais viram uma probabilidade?</h2>
            <p>O modelo final transforma o histórico em uma classificação binária de risco.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="question">
          <div class="question-bar"></div>
          <div style="width:100%;">
            <div class="eyebrow">Modelo final</div>
            <div class="question-text" style="font-size:16px;">
              Regressão logística + balanceamento de classes + ponderação de observações recentes
              → probabilidade de bandeira vermelha
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### O protocolo de modelagem")
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">01</div><h4>ALVO</h4><p>Bandeira vermelha = 1 (patamar 1 ou 2); demais situações = 0.</p></div>
          <div class="impact-card"><div class="impact-icon">02</div><h4>SEM VAZAMENTO</h4><p>Cada previsão usa apenas informações disponíveis até o mês de origem.</p></div>
          <div class="impact-card"><div class="impact-icon">03</div><h4>BACKTEST</h4><p>Janela expansiva, com mínimo de 36 meses de treino e avaliação temporal.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid" style="margin-top:14px;">
          <div class="impact-card"><div class="impact-icon">04</div><h4>NORMAL DE CHUVA</h4><p>A normal climatológica usada no percentual de chuva é construída de forma expansiva, somente com o passado.</p></div>
          <div class="impact-card"><div class="impact-icon">05</div><h4>CLASSES</h4><p>A classe vermelha é menos frequente; a ponderação evita que o modelo simplesmente favoreça a classe majoritária.</p></div>
          <div class="impact-card"><div class="impact-icon">06</div><h4>RECÊNCIA</h4><p>Observações a partir de abril de 2024 recebem peso 5x, aproximando o treino do regime mais recente.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if db_ok and not df_training.empty:
        st.markdown("### Base de treinamento")
        st.caption(f"{len(df_training):,} observações disponíveis na camada refined para apoio à leitura da modelagem.")
        st.dataframe(df_training.tail(12), use_container_width=True, hide_index=True)

    st.markdown("### O que a avaliação precisa responder")
    st.info("A qualidade do modelo deve ser lida principalmente pela capacidade de identificar meses vermelhos sem usar o futuro. Acurácia, precisão, recall e F1 ajudam a interpretar esse equilíbrio.")

# ------------------------------------------------------------
# TAB 7 — PREVISÃO
# ------------------------------------------------------------
with tabs[6]:
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-title">
            <h2>O que o modelo estima para frente?</h2>
            <p>Previsão calculada a partir da referência de {period_label(reference_month)}.</p>
          </div>
          <div class="update">Referência<br><strong>{period_label(reference_month)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reference_available and forecast_results:
        render_forecast_cards(forecast_results)
        st.success(f"Previsão executiva calculada com os dados reais disponíveis até {period_label(reference_month)} e a mesma receita do notebook 07.")
    else:
        st.warning(f"A referência {period_label(reference_month)} não está disponível nas fontes necessárias para calcular a previsão.")

    st.markdown("### Como interpretar")
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">M+1</div><h4>MAIOR VISIBILIDADE</h4><p>É o horizonte mais próximo do mês observado e, em geral, o que possui maior quantidade de informação histórica comparável.</p></div>
          <div class="impact-card"><div class="impact-icon">M+2</div><h4>MAIOR INCERTEZA</h4><p>O horizonte aumenta a distância entre a informação observada e o evento que queremos antecipar.</p></div>
          <div class="impact-card"><div class="impact-icon">M+3</div><h4>SINAL DE TENDÊNCIA</h4><p>É útil como sinal de risco, mas deve ser interpretado com mais cautela do que M+1.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# TAB 2 — METODOLOGIA
# ------------------------------------------------------------
with tabs[1]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Metodologia e limites da leitura</h2>
            <p>O resultado é uma estimativa probabilística, não uma certeza sobre o futuro.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">01</div><h4>DADOS</h4><p>ANEEL, ONS e INMET fornecem as séries utilizadas para construir o painel mensal.</p></div>
          <div class="impact-card"><div class="impact-icon">02</div><h4>MODELAGEM</h4><p>Regressão logística, engenharia de atributos e validação temporal preservam a ordem cronológica.</p></div>
          <div class="impact-card"><div class="impact-icon">03</div><h4>SAÍDA</h4><p>Probabilidade estimada de bandeira vermelha para M+1, M+2 e M+3.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Limitações")
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card"><div class="impact-icon">•</div><h4>BASE HISTÓRICA</h4><p>A série mensal é relativamente pequena e a classe vermelha é menos frequente, o que limita a complexidade do modelo.</p></div>
          <div class="impact-card"><div class="impact-icon">•</div><h4>CLIMA FUTURO</h4><p>A previsão utiliza dados observados do mês de referência; não incorpora uma previsão meteorológica futura.</p></div>
          <div class="impact-card"><div class="impact-icon">•</div><h4>PROBABILIDADE</h4><p>Uma probabilidade elevada representa sinal de risco, não garantia de ocorrência da bandeira.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(closing_section(), unsafe_allow_html=True)



def render_f1_notebook05_chart():
    """Gráfico de barras agrupadas: F1-macro de CV do notebook 05 (persistência vs. modelos aprendidos)."""
    horizons = ["t+1", "t+2", "t+3"]
    candidates = [
        ("Persistência", [0.7907, 0.7076, 0.6117], "#16a0ff"),
        ("v5: receita fixa por horizonte", [0.7818, 0.5359, 0.4881], "#3d5b74"),
        ("Regressão ordinal + CMO", [0.5758, 0.5178, 0.4151], "#3d5b74"),
        ("Controle nominal + CMO", [0.5605, 0.4854, 0.4151], "#3d5b74"),
        ("v3: clima + bandeira + ONI", [0.4611, 0.4567, 0.4063], "#3d5b74"),
        ("Classe majoritária", [0.2500, 0.2500, 0.2557], "#26384a"),
    ]
    fig = go.Figure()
    for name, values, color in candidates:
        fig.add_trace(go.Bar(
            name=name,
            x=horizons,
            y=values,
            marker_color=color,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(size=10, color="#8fa8bb"),
        ))
    fig.update_layout(
        barmode="group",
        height=370,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#07131f",
        plot_bgcolor="#07131f",
        font=dict(color="#cfe0eb", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9.5)),
        xaxis=dict(showgrid=False, color="#8fa8bb"),
        yaxis=dict(title="F1-macro (CV)", gridcolor="#183042", zeroline=False, color="#8fa8bb", range=[0, 0.95]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------
# TAB 3 — JORNADA DO PROJETO (making-of / storytelling)
# ------------------------------------------------------------
with tabs[2]:
    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como chegamos até aqui</h2>
            <p>Bastidores do projeto: notebooks testados, decisões técnicas e caminhos abandonados.</p>
          </div>
        </div>

        <div class="question">
          <div class="question-bar"></div>
          <div style="width:100%;">
            <div class="eyebrow">Por que contar essa história</div>
            <div class="question-text" style="font-size:14px;">
              Um projeto de dados não nasce pronto do jeito que aparece nas abas anteriores. O grupo
              (Alberto, Fabio, Laiane, Sweeli e Tatiane) começou em outro tema, com outras ferramentas,
              e mudou de rumo mais de uma vez. Esta seção documenta o caminho completo — da ideia
              original de energia solar até a versão validada do modelo — o que funcionou, o que foi
              abandonado e por quê, para dar transparência ao número final apresentado.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>O diagrama original da arquitetura</h2>
            <p>A arquitetura "Previsão de Bandeira Vermelha" desenhada pelo grupo, ponto de partida de toda a implementação.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    architecture_image = Path(__file__).resolve().parent / "arquitetura_previsao_bandeira_vermelha.png"
    if architecture_image.exists():
        st.image(str(architecture_image), use_container_width=True)
        st.caption("Diagrama de arquitetura desenhado pelo grupo na fase de planejamento do projeto.")
    else:
        st.warning(
            "Imagem de arquitetura não encontrada. Verifique se "
            "'arquitetura_previsao_bandeira_vermelha.png' está na pasta dashboards."
        )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como os dados chegam ao produto</h2>
            <p>Do dado bruto às tabelas utilizadas na modelagem e no consumo analítico.</p>
          </div>
        </div>

        <style>
          .pipeline-wrap { display:flex; align-items:stretch; gap:0; margin:18px 0 6px 0; flex-wrap:wrap; }
          .pipeline-col { flex:1 1 190px; min-width:170px; background:#0d1e2e; border:1px solid #1c3348;
                          border-radius:10px; padding:14px 12px; display:flex; flex-direction:column; gap:8px; }
          .pipeline-col-title { font-size:10.5px; font-weight:700; letter-spacing:.07em; text-transform:uppercase;
                                 color:#8fa8bb; margin-bottom:2px; }
          .pipeline-box { background:#132638; border-radius:6px; padding:8px 10px; font-size:12px; line-height:1.4;
                           color:#cfe0eb; border-left:3px solid var(--blue); }
          .pipeline-box b { color:#fff; }
          .pipeline-box.stage { border-left-color:var(--orange); }
          .pipeline-box.raw { border-left-color:#8fa8bb; }
          .pipeline-box.trusted { border-left-color:#ffcf4d; }
          .pipeline-box.refined { border-left-color:var(--blue-2); }
          .pipeline-arrow { display:flex; align-items:center; justify-content:center; color:#4c6a80;
                             font-size:22px; flex:0 0 26px; }
          @media (max-width:900px) {
            .pipeline-wrap { flex-direction:column; }
            .pipeline-arrow { transform:rotate(90deg); padding:2px 0; }
          }
        </style>

        <div class="pipeline-wrap">
          <div class="pipeline-col">
            <div class="pipeline-col-title">Fontes de dados</div>
            <div class="pipeline-box"><b>APIs meteorológicas</b><br>INMET / CPTEC</div>
            <div class="pipeline-box"><b>Dados operacionais</b><br>ONS — ENA, EAR, CMO, carga</div>
            <div class="pipeline-box"><b>Histórico regulatório</b><br>ANEEL — bandeiras tarifárias</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-col">
            <div class="pipeline-col-title">Orquestração &amp; ingestão</div>
            <div class="pipeline-box"><b>Scripts de ingestão</b><br><code>scripts/01_ingestao_stage</code></div>
            <div class="pipeline-box"><b>Versionamento</b><br>Databricks Repos ↔ GitHub, branch <code>desenv</code></div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-col">
            <div class="pipeline-col-title">Data lake (Databricks)</div>
            <div class="pipeline-box stage"><b>Stage</b> — dados brutos ingeridos, limpeza básica</div>
            <div class="pipeline-box raw"><b>Raw</b> — tipagem e padronização</div>
            <div class="pipeline-box trusted"><b>Trusted</b> — dados validados e estruturados</div>
            <div class="pipeline-box refined"><b>Refined</b> — tabelas prontas para consumo</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-col">
            <div class="pipeline-col-title">Consumo de dados</div>
            <div class="pipeline-box"><b>Modelagem</b><br><code>scripts/05_modelagem</code>, notebooks 01–07</div>
            <div class="pipeline-box"><b>Streamlit</b><br>este painel e o portal operacional</div>
            <div class="pipeline-box"><b>Relatórios</b><br>previsão executiva por horizonte</div>
          </div>
        </div>
        <p style="font-size:11px; color:#63788a; margin-top:8px;">
          Versão adaptada à estrutura efetivamente implementada no repositório
          <a href="https://github.com/FabioFumioWada/MBA_Eng_Dados_TurmaG_Energia_Solar/tree/desenv" target="_blank">MBA_Eng_Dados_TurmaG_Energia_Solar</a>
          (branch <code>desenv</code>).
        </p>

        <div class="impact-grid" style="margin-top:14px;">
          <div class="impact-card">
            <div class="impact-icon">◇</div>
            <h4>INMET &amp; CPTEC</h4>
            <p>APIs públicas do Instituto Nacional de Meteorologia e do CPTEC/INPE — chuva, temperatura,
            umidade e o índice El Niño (ONI) usados como sinal climático mensal.</p>
          </div>
          <div class="impact-card">
            <div class="impact-icon">◇</div>
            <h4>ONS</h4>
            <p>API/portal de dados abertos do Operador Nacional do Sistema Elétrico — ENA, EAR, CMO e
            carga, atualizados mensalmente por subsistema.</p>
          </div>
          <div class="impact-card">
            <div class="impact-icon">◇</div>
            <h4>ANEEL</h4>
            <p>Histórico oficial das bandeiras tarifárias — a variável que o modelo tenta prever, também
            consumida via dados públicos do regulador.</p>
          </div>
        </div>
        <p style="font-size:11px; color:#63788a; margin-top:10px;">
          As três são APIs/portais públicos e gratuitos — nenhuma credencial paga ou acesso restrito é
          necessário, o que manteve o pipeline reprodutível para qualquer integrante do grupo rodar do zero.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Quatro fases até o modelo atual</h2>
            <p>O projeto trocou de tema, de escopo e de abordagem de modelagem antes de chegar à versão em produção.</p>
          </div>
        </div>

        <div class="impact-grid">
          <div class="impact-card abandoned">
            <div class="impact-icon">F1</div>
            <h4>FASE 1 · ESCOPO ORIGINAL</h4>
            <p>"Energia em Dados: Geração Solar, Tarifas e Reservatórios no Brasil". Geração distribuída
            solar, tarifas TE/TUSD, ENA por bacia e clusterização de UF, com dashboard em Power BI.
            Registrado no repositório <code>MACK_MBA_Eng_Dados_TurmaG_Energia_Solar</code>.</p>
          </div>
          <div class="impact-card abandoned">
            <div class="impact-icon">F2</div>
            <h4>FASE 2 · SPRINT 2</h4>
            <p>Escopo reduzido para o Brasil como um todo, com a solar ainda central: a janela de 30 a
            120 dias para instalar um sistema solar vira a métrica de decisão do consumidor. Camadas
            Raw/Trusted/Refined chegaram a ser criadas no Databricks, mas o push para o GitHub nunca
            foi concluído.</p>
          </div>
          <div class="impact-card abandoned">
            <div class="impact-icon">01-06</div>
            <h4>MODELAGEM MULTICLASSE</h4>
            <p>Seis notebooks depois, com clima e hidrologia (ENA/EAR/CMO/El Niño) no lugar da solar, o
            resultado foi honesto e desconfortável: o modelo de persistência (repetir a última bandeira)
            venceu todos os modelos treinados, em todos os horizontes, em validação cruzada. Conduzida
            em <code>scripts/05_modelagem</code> — a mesma pasta, no repositório sem o prefixo
            MACK, onde depois nasceria o notebook 07.</p>
          </div>
          <div class="impact-card done">
            <div class="impact-icon">07</div>
            <h4>MODELO EM PRODUÇÃO</h4>
            <p>Virada de chave: classificação binária (vermelha ou não) com régua de marcos regulatórios
            e peso 5x para dados recentes. É a única tentativa, entre todas, que superou a persistência
            — e é o que alimenta esta aplicação.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Linha do tempo do desenvolvimento</h2>
            <p>Da ideia original de energia solar ao modelo binário em produção.</p>
          </div>
        </div>

        <div class="timeline">

          <div class="tl-item">
            <div class="tl-dot abandoned"></div>
            <div class="tl-tag abandoned">Fase 1 · 15/08/2026</div>
            <div class="tl-title">Projeto começa com foco em energia solar</div>
            <div class="tl-text">
              Escopo original: relação entre a expansão da geração distribuída solar, as tarifas TE/TUSD
              das distribuidoras e o nível dos reservatórios (ENA), com clusterização de estados por
              perfil de penetração solar e um dashboard em Power BI / Looker Studio. Estrutura registrada
              no repositório
              <a href="https://github.com/FabioFumioWada/MACK_MBA_Eng_Dados_TurmaG_Energia_Solar" target="_blank">MACK_MBA_Eng_Dados_TurmaG_Energia_Solar</a>.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot warn"></div>
            <div class="tl-tag warn">Fase 2 · Sprint 2</div>
            <div class="tl-title">Escopo nacional e a janela de decisão do consumidor</div>
            <div class="tl-text">
              A pergunta de negócio foi reformulada: além de prever a bandeira, mostrar o quanto a
              energia solar mitigaria o impacto — aproveitando que instalar um sistema residencial leva
              de 30 a 120 dias, quase um trimestre de antecedência. Camadas Raw/Trusted/Refined chegaram
              a ser criadas no workspace do Databricks para o mesmo repositório, mas o commit e push para
              o GitHub nunca foram concluídos por integração externa indisponível na sessão.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot abandoned"></div>
            <div class="tl-tag abandoned">Notebooks 01–06</div>
            <div class="tl-title">Resultado honesto: a persistência venceu</div>
            <div class="tl-text">
              A energia solar saiu do modelo. O grupo conduziu seis notebooks testando classificação
              multiclasse (Verde / Amarela / Vermelha) com clima, ENA, EAR, CMO e índice El Niño (ONI),
              dentro de <code>scripts/05_modelagem</code> no repositório
              <a href="https://github.com/FabioFumioWada/MBA_Eng_Dados_TurmaG_Energia_Solar" target="_blank">MBA_Eng_Dados_TurmaG_Energia_Solar</a>
              (sem o prefixo MACK) — a mesma pasta onde, mais tarde, nasceria o notebook 07. Com validação
              temporal rigorosa (TimeSeriesSplit com gap, métricas OOF, F1-macro), em todos os horizontes
              o modelo de persistência — simplesmente repetir a última bandeira — superou todos os
              modelos treinados. O resultado negativo foi documentado e mantido, sem maquiagem, para a
              banca.
            </div>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="chart-card" style="margin:4px 0 18px 30px;">
          <div class="chart-title">Notebook 05 — a persistência venceu em todos os horizontes</div>
          <div class="chart-sub">F1-macro de validação cruzada (CV), previsões OOF, calendário comum de 45 meses
          por horizonte · Fonte: Relatório Consolidado das Modelagens, notebook 05
          (resultado_databricks.json, campo ResumoCV), corte em 08/09/2026.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_f1_notebook05_chart()

    st.markdown(
        """
        <div class="timeline" style="margin-top:0;">

          <div class="tl-item">
            <div class="tl-dot done"></div>
            <div class="tl-tag done">Virada</div>
            <div class="tl-title">Simplificar para binário e usar conhecimento de domínio</div>
            <div class="tl-text">
              Diante do resultado anterior, o grupo trocou o problema de multiclasse para binário
              (vermelha ou não) e passou a incorporar conhecimento de domínio: uma régua de marcos
              regulatórios e peso 5x para observações recentes. Essa combinação foi a única, entre todas
              as tentativas, a superar a persistência — e deu origem ao notebook 07.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot"></div>
            <div class="tl-tag">Ponto de partida</div>
            <div class="tl-title">Arquitetura em camadas já definida</div>
            <div class="tl-text">
              Stage → Raw → Trusted → Refined → Modelagem, com três variantes de notebook de
              regressão logística (07 principal, 07_P e 07_PP) alimentando aplicações diferentes do grupo.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot"></div>
            <div class="tl-tag">Diagnóstico</div>
            <div class="tl-title">"Por que o notebook 07 não avança para os meses mais recentes?"</div>
            <div class="tl-text">
              Investigação mostrou que não era um bug: o backtest só avalia meses cuja bandeira
              oficial já existe em <code>mba.trusted.f_bandeira</code>. É uma proteção contra
              vazamento de dados (anti-leakage), não uma falha de código.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot done"></div>
            <div class="tl-tag done">Correção de dados</div>
            <div class="tl-title">Defasagem de publicação da ANEEL</div>
            <div class="tl-text">
              A ANEEL antecipa a divulgação da bandeira do mês seguinte — setembro/2026 já havia
              sido publicada em 28/08 como amarela. A base foi reprocessada para ligar o clima de
              agosto à competência de setembro, deslocando a fronteira real de previsão para
              outubro em diante.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot warn"></div>
            <div class="tl-tag warn">Governança de dados</div>
            <div class="tl-title">Padronização do fluxo de versionamento entre integrantes</div>
            <div class="tl-text">
              O grupo formalizou boas práticas de versionamento — <code>git status</code> /
              <code>git add</code> explícitos antes de cada commit, com <code>pull</code> cuidadoso
              nos clones de cada integrante — para garantir consistência do histórico do repositório
              ao longo da colaboração.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot abandoned"></div>
            <div class="tl-tag abandoned">Testado e abandonado</div>
            <div class="tl-title">Dashboard antigo só de EDA</div>
            <div class="tl-text">
              O <code>dashboards/streamlit_app.py</code> criado na Fase 2 — dentro do repositório
              MACK_MBA — lia três parquets (<code>solar_mensal_nacional</code>, <code>ena_mensal</code>,
              <code>bandeira_mensal</code>) que eram apenas ponteiros de Git LFS dentro do Databricks
              Repos — as abas de Solar, ENA e Bandeiras ficavam em branco, e um <code>try/except</code>
              escondia o erro em vez de expô-lo. Não havia aba de previsão nem de acurácia, insuficiente
              para contar a história completa do projeto.
            </div>
          </div>

          <div class="tl-item">
            <div class="tl-dot done"></div>
            <div class="tl-tag done">Estado atual</div>
            <div class="tl-title">Notebook 07 principal validado e em produção</div>
            <div class="tl-text">
              Régua de 4 marcos regulatórios, peso 5x para observações recentes, backtest expansivo
              com intervalo de confiança de Wilson. Esta aplicação consulta os mesmos dados reais do
              Databricks usados pelo notebook para gerar o painel executivo e a previsão.
            </div>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Onde você vai ver o modelo funcionando</h2>
            <p>O mesmo notebook de regressão logística alimenta três aplicações — duas delas serão demonstradas ao vivo na apresentação.</p>
          </div>
        </div>

        <div class="impact-grid">
          <div class="impact-card done">
            <div class="impact-icon">07</div>
            <h4>PRINCIPAL · ESTE PAINEL</h4>
            <p>Versão de referência com a régua regulatória completa. É a que alimenta esta aplicação e a aba de Previsão mostrada nesta jornada.</p>
          </div>
          <div class="impact-card done">
            <div class="impact-icon">P</div>
            <h4>07_P · PORTAL OPERACIONAL</h4>
            <p>Envia ano/mês ao notebook no Databricks e devolve a previsão em JSON para os três horizontes (t+1, t+2, t+3).</p>
            <p style="margin-top:6px;"><a href="https://previsao-bandeira.streamlit.app/" target="_blank">Abrir portal ↗</a></p>
          </div>
          <div class="impact-card done">
            <div class="impact-icon">PP</div>
            <h4>07_PP · PORTAL DE PESQUISA</h4>
            <p>Recebe cinco parâmetros meteorológicos digitados pelo usuário e retorna a previsão parametrizada para o horizonte t+1.</p>
            <p style="margin-top:6px;"><a href="https://previsao-bandeira-pesquisa.streamlit.app/" target="_blank">Abrir portal ↗</a></p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Ferramentas e pilha técnica</h2>
            <p>O que sustenta o projeto do dado bruto até esta tela.</p>
          </div>
        </div>
        <div>
          <span class="tool-pill">● Databricks · Workspace &amp; Notebooks</span>
          <span class="tool-pill">● Databricks · Repos / Git Folder</span>
          <span class="tool-pill">● Unity Catalog</span>
          <span class="tool-pill">● SQL Warehouse</span>
          <span class="tool-pill">◆ GitHub · branch desenv</span>
          <span class="tool-pill">◆ Git LFS</span>
          <span class="tool-pill">● Python</span>
          <span class="tool-pill">● PySpark</span>
          <span class="tool-pill">● pandas / NumPy</span>
          <span class="tool-pill">● scikit-learn · Regressão logística</span>
          <span class="tool-pill">● Plotly</span>
          <span class="tool-pill">● Streamlit · dois apps distintos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Ferramentas testadas e substituídas</h2>
            <p>Pilha planejada nas fases 1 e 2, antes da migração para o fluxo atual em Databricks.</p>
          </div>
        </div>
        <div>
          <span class="tool-pill">◆ Power BI (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ Google Looker Studio (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ K-Means · clusterização de UF (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ DuckDB (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ Prefect / cron (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ Trello / Jira (Fase 1 · descontinuado)</span>
          <span class="tool-pill">◆ ANEEL · Geração Distribuída Solar (Fase 1-2 · descontinuado)</span>
          <span class="tool-pill">◆ Índice ONI / El Niño (Notebooks 01–06 · descontinuado)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="closing" style="margin-top:26px;">
          <div class="quote">"</div>
          <div class="closing-text">
            O objetivo desta aba não é esconder as tentativas que não deram certo — da energia solar
            abandonada em MACK_MBA à persistência que venceu seis notebooks de modelagem.
            É mostrar o raciocínio — e as correções de rota — que sustentam o número final.
            <small>SOLAR · MULTICLASSE · VIRADA · VALIDAÇÃO</small>
          </div>
          <div class="closing-side">
            Solar<br>
            Multiclasse<br>
            Virada<br>
            Validação
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )