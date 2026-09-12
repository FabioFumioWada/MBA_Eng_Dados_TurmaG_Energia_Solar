# Databricks notebook source
# MAGIC %pip install streamlit pandas numpy scikit-learn

# COMMAND ----------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Simulador - Bandeira Vermelha", layout="wide")

st.title("Simulador de Previsão de Bandeira Vermelha")
st.markdown("Camada de Consumo: Interpretação do impacto do clima e armazenamento.")

@st.cache_resource
def treinar_modelo_simulado():
    """Treina o modelo idêntico ao seu script usando dados sintéticos para demonstração."""
    cols = ["mes_clima", "chuva_media", "chuva_pct_normal_ok", "temperatura", 
            "umidade", "bandeira_origem", "ear_pct", "regime_gsf_pld", 
            "regime_faixas_2019", "intervencao_pandemia", "intervencao_escassez"]
    
    rng = np.random.default_rng(42)
    # Gerando 500 linhas sintéticas
    X_mock = pd.DataFrame({
        "mes_clima": rng.integers(1, 13, 500),
        "chuva_media": rng.uniform(0, 300, 500),
        "chuva_pct_normal_ok": rng.uniform(50, 150, 500),
        "temperatura": rng.uniform(18, 35, 500),
        "umidade": rng.uniform(40, 90, 500),
        "bandeira_origem": rng.choice([0, 1], 500),
        "ear_pct": rng.uniform(20, 80, 500),
        "regime_gsf_pld": rng.choice([1], 500),
        "regime_faixas_2019": rng.choice([1], 500),
        "intervencao_pandemia": rng.choice([0, 1], 500, p=[0.9, 0.1]),
        "intervencao_escassez": rng.choice([0, 1], 500, p=[0.9, 0.1]),
    })
    
    # Criando um alvo onde temperatura alta e chuva/ear baixos = Bandeira Vermelha
    z = (X_mock["temperatura"] * 0.3) - (X_mock["chuva_pct_normal_ok"] * 0.1) - (X_mock["ear_pct"] * 0.15)
    prob = 1 / (1 + np.exp(-z))
    y_mock = (prob > np.median(prob)).astype(int)
    
    # O pipeline exato do seu código
    clf = LogisticRegression(max_iter=5000, random_state=42)
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", clf)
    ])
    model.fit(X_mock, y_mock)
    
    # Extraindo coeficientes para explicação
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=cols)
    return model, cols, coefs

modelo, colunas_features, coeficientes = treinar_modelo_simulado()

# --- BARRA LATERAL (INPUTS) ---
st.sidebar.header("Parâmetros de Simulação (Mês Alvo)")

temp = st.sidebar.slider("Temperatura Média (°C)", 15.0, 40.0, 25.0)
chuva_pct = st.sidebar.slider("Chuva (% da Normal)", 10.0, 200.0, 100.0)
ear = st.sidebar.slider("EAR Subsistema SE (%)", 10.0, 100.0, 50.0)

st.sidebar.markdown("---")
chuva_media = st.sidebar.slider("Chuva Média (mm)", 0.0, 400.0, 150.0)
umidade = st.sidebar.slider("Umidade Média (%)", 30.0, 100.0, 70.0)
band_ant = st.sidebar.selectbox("Bandeira Mês Anterior", [0, 1], format_func=lambda x: "Vermelha" if x == 1 else "Outra")

# --- INFERÊNCIA ---
input_df = pd.DataFrame([[
    6, chuva_media, chuva_pct, temp, umidade, band_ant, ear,
    1, 1, 0, 0 # Marcos regulatórios recentes
]], columns=colunas_features)

probabilidade = modelo.predict_proba(input_df)[0][1]

# --- PAINEL PRINCIPAL ---
st.subheader("Resultado da Previsão")
cor = "red" if probabilidade > 0.5 else "green"
st.markdown(f"<h1 style='color: {cor};'>{probabilidade:.1%} de chance de Bandeira Vermelha</h1>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Interpretação do Modelo (Pesos Climáticos)")
st.write("Abaixo estão os pesos que o algoritmo atribuiu às variáveis climáticas (valores padronizados pelo `StandardScaler`). Valores positivos aumentam a chance de bandeira vermelha, valores negativos reduzem.")

st.bar_chart(coeficientes[["temperatura", "chuva_pct_normal_ok", "ear_pct", "umidade"]])