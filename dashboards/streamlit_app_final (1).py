
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

    st.markdown("### Simulador de cenários")
    st.caption("Altere as condições observadas na referência e veja como a mesma receita de modelagem responde. A simulação é uma análise de sensibilidade; a previsão executiva acima usa os valores reais.")

    if reference_available:
        ref_row = get_reference_row(sim_base, reference_month)
        with st.form("form_simulador_final"):
            c1, c2, c3 = st.columns(3)
            with c1:
                temperatura = st.number_input("Temperatura média (°C)", value=float(ref_row["temperatura"]) if pd.notna(ref_row["temperatura"]) else 25.0, step=0.5)
                chuva_media = st.number_input("Chuva média (mm)", value=float(ref_row["chuva_media"]) if pd.notna(ref_row["chuva_media"]) else 100.0, step=5.0)
                chuva_acum = st.number_input("Chuva acumulada (mm)", value=float(ref_row["chuva_acum"]) if pd.notna(ref_row["chuva_acum"]) else 100.0, step=5.0)
            with c2:
                chuva_pct = st.number_input("Chuva (% da normal)", value=float(ref_row["chuva_pct_normal_ok"]) if pd.notna(ref_row["chuva_pct_normal_ok"]) else 100.0, step=5.0)
                umidade = st.number_input("Umidade média (%)", value=float(ref_row["umidade"]) if pd.notna(ref_row["umidade"]) else 70.0, step=1.0)
            with c3:
                ear = st.number_input("EAR SE (%)", value=float(ref_row["ear_pct"]) if pd.notna(ref_row["ear_pct"]) else 50.0, step=1.0)
                band_ant = st.selectbox("Bandeira anterior", [0, 1], index=int(float(ref_row["bandeira_origem"])) if pd.notna(ref_row["bandeira_origem"]) else 0, format_func=lambda x: "Vermelha" if x == 1 else "Não vermelha")
            submitted = st.form_submit_button("▶ SIMULAR CENÁRIO", use_container_width=True)

        if submitted:
            scenario = {
                "mes_clima": int(reference_month.month),
                "chuva_acum": chuva_acum,
                "chuva_media": chuva_media,
                "chuva_pct_normal_ok": chuva_pct,
                "temperatura": temperatura,
                "umidade": umidade,
                "bandeira_origem": band_ant,
                "ear_pct": ear,
            }
            sim_results = []
            for h in (1, 2, 3):
                try:
                    p, target, n_train = predict_scenario(sim_base, sim_series, reference_month, scenario, h)
                    sim_results.append({"h": h, "prob": p, "target": target, "n_train": n_train})
                except Exception as exc:
                    sim_results.append({"h": h, "prob": None, "target": reference_month + h, "n_train": None, "error": str(exc)})
            render_forecast_cards(sim_results, title="Probabilidade no cenário simulado")
            st.info("A simulação altera apenas as entradas. Ela não altera a previsão executiva nem os dados do Databricks.")
    else:
        st.info("O simulador ficará disponível quando a referência de agosto estiver presente nas três fontes utilizadas pelo modelo.")

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
        <div style="background:#fff; border-radius:12px; padding:10px; margin-top:6px;">
          <img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAKIA7UDASIAAhEBAxEB/8QAHQAAAQQDAQEAAAAAAAAAAAAAAAEFBgcCAwQICf/EAG4QAAEDBAADAwYHCAwHCwcICwECAwQABQYRBxIhEzFBCBQiUWFxFRYyM4GV0jdTVHWRsrPTFyM0NkJScnaTobHRCSRXYpKUtCYnQ1ZldIKio8HCJUZHVXOEhRg1REVjZIOWw+HwKClmpDjj8dT/xAAbAQEBAQEBAQEBAAAAAAAAAAAAAQIDBAUGB//EADoRAQEAAgEEAQMBBwIEBQQDAAABAhEDBBIhMQUTQVEGFBUWIjJSYSNxQlOBkQczNMHwJLHR4UOh8f/aAAwDAQACEQMRAD8A9l0UVrkPIYZU6sOFKe8NtqWrv8AkEmg2UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8MRPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD6vf+xR8LxPvVw+r3/sUDhRTf8LxPvVw+r3/sUfC8T71cPq9/7FA4UU3/AAvE+9XD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ffr2b+toHyjVMfxUtf4Vffr2b+to+Klr/Cr79ezf1tA+ao1TH8VLX+FX369m/raPipa/wq+/Xs39bQPn0UvhTF8VLX+FX369m/raPipa/wAKvv17N/W0D5R9FMfxUtf4Vffr2b+to+Klr/Cr79ezf1tA+UGmP4qWv8Kvv17N/W0fFS1/hV9+vZv62gfNUH6KY/ipa/wq+/Xs39bR8VLX+FX369m/raB88aKY/ipa/wAKvv17N/W0fFS1/hV9+vZv62gfKKY/ipa/wq+/Xs39bR8VLX+FX369m/raB8FGqY/ipa/wq+/Xs39bR8VLX+FX369m/raB8FFMfxUtf4Vffr2b+to+Klr/AAq+/Xs39bQPopNUx/FS1/hV9+vZv62j4qWv8Kvv17N/W0D51o60x/FO1/hV++vZv62j4qWv8Kv317N/W0D51o60x/FS1/hV9+vZv62j4qWv8Kvv17N/W0D5R13TH8VLX+FX369m/raPipa/wq+/Xs39bQPh3RTH8VLX+FX369m/raPipa/wq+/Xs39bQPlHWmP4qWv8Kvv17N/W0fFS1/hV9+vZv62gfDR1pj+Klr/Cr79ezf1tHxUtf4Vffr2b+toH3wpNUx/FS1/hV9+vZv62j4qWv8Kvv17N/W0D5R1pj+Klr/Cr79ezf1tHxUtf4Vffr2b+toHzrRTH8VLX+FX769m/raPipa/wq+/Xs39bQPlH0Ux/FS1/hV9+vZv62j4qWv8ACr79ezf1tA+aopj+Klr/AAq+/Xs39bR8VLX+FX369m/raB8opj+Klr/Cr79ezf1tHxUtf4Vffr2b+toHyimP4qWv8Kvv17N/W0fFS1/hV9+vZv62gfDRrpTH8VLX+FX369m/raPipa/wq+/Xs39bQPmjRTH8VLX+FX369m/raPipa/wq+/Xs39bQPmqKY/ipa/wq+/Xs39bR8VLX+FX369m/raB8+ig+6mP4qWv8Kvv17N/W0fFS1/hV++vZv62gfBR9FMfxUtf4Vffr2b+to+Klr/Cr79ezf1tA+H3UUx/FS1/hV9+vZv62j4qWv8Kvv17N/W0D53eqj8lMfxUtf4Vffr2b+to+Klr/AAq+/Xs39bQPmqDTH8VLX+FX369m/raPipa/wq+/Xs39bQPmqKY/ipa/wq+/Xs39bR8VLX+FX369m/raB8o/JTH8VLX+FX369m/raPipa/wq+/Xs39bQPn5KPyUx/FS1/hV9+vZv62j4qWv8Kvv17N/W0D51o678KY/ipa/wq+/Xs39bR8VLX+FX369m/raB86+qjVMfxUtf4Vfvr2b+to+Klr/Cr79ezf1tA+0Uzw8ct8SU3JakXhS2zsB28SnUH3pW4Un3EGnigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKAooooCiiigKKKKD//2Q==" style="width:100%; display:block; border-radius:6px;" />
        </div>
        <p style="font-size:11px; color:#63788a; margin-top:8px;">
          Diagrama de arquitetura desenhado pelo grupo na fase de planejamento do projeto.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
          <div class="section-title">
            <h2>Como o grupo implementou essa arquitetura</h2>
            <p>A estrutura de dados efetivamente implementada, do dado bruto até o consumo pelas aplicações do grupo.</p>
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
            <div class="tl-tag warn">Risco operacional</div>
            <div class="tl-title">Painel "Commit &amp; Push" do Databricks apagou arquivos rastreados</div>
            <div class="tl-text">
              O painel indicava "No changed files" e ainda assim causou perda de conteúdo já
              versionado. O fluxo foi trocado para <code>git status</code> / <code>git add</code>
              explícito via terminal, com <code>pull</code> cuidadoso nos clones dos demais integrantes.
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
