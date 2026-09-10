# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Regressao Logistica com Sazonalidade e Clima (regua de 4 marcos, versao final corrigida)
# MAGIC Modelo de comparacao (junto do notebook 07) para prever a bandeira tarifaria vermelha
# MAGIC nos horizontes t+1, t+2 e t+3, com o protocolo unificado de teste.
# MAGIC
# MAGIC Esta e a versao final desta especificacao de sazonalidade e clima (a mesma familia de variaveis do notebook 02), mas
# MAGIC com as tres correcoes que o notebook 02 de producao ainda nao tem:
# MAGIC - Regua de 4 marcos regulatorios em vez dos 5 marcadores atuais (exclui de proposito o
# MAGIC   abaco de 2024, para nao contaminar o teste com uma variavel muito perto do periodo
# MAGIC   avaliado).
# MAGIC - Normal de chuva corrigida (calculada de forma expansiva, media dos mesmos meses do
# MAGIC   calendario ocorridos antes de cada mes, nunca usando informacao futura) em vez da
# MAGIC   normal com vazamento usada hoje.
# MAGIC - Teste em backtest longo (janela expansiva desde jan/2019, treino minimo de 36 meses)
# MAGIC   em vez do teste curto de 17 meses (abr/2025 a ago/2026) usado hoje.
# MAGIC O peso 5x para o periodo pos abaco (abr/2024) e a reponderacao de classes ja existiam no
# MAGIC notebook 02 de producao e sao mantidos aqui.

# COMMAND ----------

import io, sys
_buf = io.StringIO(); _old = sys.stdout; sys.stdout = _buf

# COMMAND ----------

import math
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Carga dos dados reais do Unity Catalog
# ---------------------------------------------------------------------------
print("[1] lendo mba.refined.f_modelo_bandeira_clima (clima INMET real)")
clima = spark.table("mba.refined.f_modelo_bandeira_clima").toPandas()
print("    linhas:", len(clima))

print("[2] lendo mba.trusted.f_bandeira (serie oficial do alvo)")
band = spark.table("mba.trusted.f_bandeira").toPandas()
print("    linhas:", len(band), "| periodo:", band.MesCompetencia.min(), "a", band.MesCompetencia.max())

b = clima.sort_values("MesCompetencia").reset_index(drop=True).copy()
b = b.rename(columns={"MesReferenciaClima": "MesRef"})
serie_band = band.sort_values("MesCompetencia").set_index("MesCompetencia")["IsVermelha"]


def mes_menos(aaaamm, k):
    a, m = aaaamm // 100, aaaamm % 100
    t = a * 12 + (m - 1) - k
    return (t // 12) * 100 + (t % 12) + 1


b["IsVermelha"] = [int(serie_band.get(m, np.nan)) if m in serie_band.index else np.nan
                   for m in b.MesCompetencia]
b["IsVermelhaMesAnterior"] = [float(serie_band.get(mes_menos(m, 1), np.nan))
                              for m in b.MesCompetencia]

bruto = b.dropna(subset=["IsVermelha", "IsVermelhaMesAnterior"]).reset_index(drop=True)
bruto["IsVermelha"] = bruto["IsVermelha"].astype(int)
print("\n[base montada] linhas:", len(bruto),
      "| periodo:", bruto.MesCompetencia.min(), "a", bruto.MesCompetencia.max(),
      "| vermelhas:", int(bruto.IsVermelha.sum()))


def para_periodo(aaaamm):
    aaaamm = int(aaaamm)
    return pd.Period(year=aaaamm // 100, month=aaaamm % 100, freq="M")


bruto["origem"] = bruto["MesRef"].map(para_periodo)
bruto["alvo_t1"] = bruto["MesCompetencia"].map(para_periodo)
serie_bandeira = pd.Series(bruto["IsVermelha"].values, index=bruto["alvo_t1"].values).sort_index()

FEATURES_ORIGEM = {
    "chuva_acum": "PrecipitacaoAcumuladaMm",
    "temperatura": "TemperaturaMediaC",
    "umidade": "UmidadeMediaPct",
    "bandeira_origem": "IsVermelhaMesAnterior",
}
base = pd.DataFrame({"origem": bruto["origem"]})
for novo, antigo in FEATURES_ORIGEM.items():
    base[novo] = bruto[antigo].values
base = base.set_index("origem").sort_index()

# ---------------------------------------------------------------------------
# 2. Correcao do vazamento na normal de chuva (so passado, nunca o futuro)
# ---------------------------------------------------------------------------
print("\n[3] corrigindo a normal de chuva (normal expansiva, so com o passado)")
base = base.sort_index()
ca = base["chuva_acum"].astype(float).values
ca3 = base["chuva_acum"].astype(float).rolling(3, min_periods=3).sum().values
mes_cal = np.array([p.month for p in base.index])
n_base = len(base)


def normal_expansiva(serie):
    """Para cada mes i, media dos mesmos meses do calendario ocorridos ANTES de i."""
    out = np.full(n_base, np.nan)
    for i in range(n_base):
        ant = [serie[j] for j in range(i) if mes_cal[j] == mes_cal[i] and not np.isnan(serie[j])]
        if ant:
            out[i] = float(np.mean(ant))
    return out


norm_exp = normal_expansiva(ca)
norm3_exp = normal_expansiva(ca3)
base["chuva_pct_normal_ok"] = np.where(norm_exp > 0, ca / norm_exp * 100, np.nan)
base["chuva_pct_normal_3m_ok"] = np.where(norm3_exp > 0, ca3 / norm3_exp * 100, np.nan)
print("    primeiro mes com normal corrigida disponivel:",
      str(base.index[int(np.argmax(~np.isnan(norm_exp)))]))

# ---------------------------------------------------------------------------
# 3. Regua de 4 marcos regulatorios (exclui de proposito o abaco de 2024)
# ---------------------------------------------------------------------------
MARCOS_4 = {
    "regime_gsf_pld": (pd.Period("2018-12", "M"), None),
    "regime_faixas_2019": (pd.Period("2019-06", "M"), None),
    "intervencao_pandemia": (pd.Period("2020-05", "M"), pd.Period("2020-11", "M")),
    "intervencao_escassez": (pd.Period("2021-09", "M"), pd.Period("2022-04", "M")),
}


def marca_regime(meses, nome):
    ini, fim = MARCOS_4[nome]
    v = meses >= ini
    if fim is not None:
        v = v & (meses <= fim)
    return v.astype(int)


COLS_MODELO = ["bandeira_origem", "sin_mes", "cos_mes", "chuva_acum", "chuva_pct_normal_ok",
               "chuva_pct_normal_3m_ok", "temperatura", "umidade"] + list(MARCOS_4)


def montar(h):
    """Uma linha por mes de origem o. Alvo = y(o+h). So usa informacao conhecida em o."""
    linhas = []
    for o, r in base.iterrows():
        alvo_mes = o + h
        if alvo_mes not in serie_bandeira.index:
            continue
        d = {"origem": o, "alvo_mes": alvo_mes, "y": int(serie_bandeira.loc[alvo_mes])}
        for c in ["bandeira_origem", "chuva_acum", "chuva_pct_normal_ok",
                  "chuva_pct_normal_3m_ok", "temperatura", "umidade"]:
            d[c] = r[c]
        m = alvo_mes.month
        d["sin_mes"] = np.sin(2 * np.pi * m / 12)
        d["cos_mes"] = np.cos(2 * np.pi * m / 12)
        linhas.append(d)
    df = pd.DataFrame(linhas)
    alvos = pd.PeriodIndex(df["alvo_mes"])
    for nome in MARCOS_4:
        df[nome] = marca_regime(alvos, nome)
    return df


# ---------------------------------------------------------------------------
# 4. Protocolo de teste: backtest longo, janela expansiva
# ---------------------------------------------------------------------------
POS_ABACO = pd.Period("2024-04", "M")
MIN_TREINO = 36
PESO_RECENTE = 5.0
INICIO_TESTE = pd.Period("2019-01", "M")


def met(y, yp):
    return dict(n=len(y), n_verm_real=int(y.sum()), n_verm_prev=int(yp.sum()),
                acuracia=accuracy_score(y, yp) * 100,
                precisao=precision_score(y, yp, zero_division=0) * 100,
                recall=recall_score(y, yp, zero_division=0) * 100,
                f1_vermelha=f1_score(y, yp, zero_division=0) * 100)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - e), 100 * min(1, c + e))


def peso_amostra(y_tr, alvos_tr):
    classes, counts = np.unique(y_tr, return_counts=True)
    n = len(y_tr)
    cw = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    w_classe = np.array([cw[v] for v in y_tr])
    w_recente = np.where(pd.PeriodIndex(alvos_tr) >= POS_ABACO, PESO_RECENTE, 1.0)
    return w_classe * w_recente


def modelo_final():
    clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("clf", clf)])


def roda(h):
    df = montar(h)
    alvos_teste = [a for a in sorted(df["alvo_mes"].unique()) if a >= INICIO_TESTE]
    cols = [c for c in COLS_MODELO if c in df.columns]
    yr, yp, meses = [], [], []
    for alvo in alvos_teste:
        lt = df[df["alvo_mes"] == alvo]
        o = lt["origem"].iloc[0]
        tr = df[df["alvo_mes"] <= o]
        if len(tr) < MIN_TREINO or tr["y"].nunique() < 2:
            continue
        usadas = [c for c in cols if tr[c].nunique(dropna=True) > 1]
        if not usadas:
            continue
        m = modelo_final()
        w = peso_amostra(tr["y"].values, tr["alvo_mes"].values)
        try:
            m.fit(tr[usadas], tr["y"], clf__sample_weight=w)
            p = int(m.predict(lt[usadas])[0])
        except Exception:
            continue
        yr.append(int(lt["y"].iloc[0])); yp.append(p); meses.append(alvo)
    return np.array(yr), np.array(yp), meses


print("\n" + "#" * 108)
print("# BACKTEST LONGO: NOTEBOOK 08 (sazonalidade e clima, regua de 4 marcos, peso 5x pos abaco, versao final)")
print("#" * 108)
resumo = []
for h in (1, 2, 3):
    y, yp, meses = roda(h)
    k = int((y == yp).sum()); n = len(y)
    lo, hi = wilson(k, n)
    m = met(y, yp)
    print(f"\n--- t+{h} ---")
    print(f"  periodo de teste: {meses[0]} a {meses[-1]}  ({n} previsoes, {int(y.sum())} vermelhas)")
    print(f"  acuracia = {100*k/n:.1f}% ({k}/{n})  IC95% (Wilson) = [{lo:.1f}% ; {hi:.1f}%]")
    print(f"  F1 vermelha = {m['f1_vermelha']:.1f}%  precisao = {m['precisao']:.1f}%  recall = {m['recall']:.1f}%")
    resumo.append({"horizonte": f"t+{h}", "n": n, "corretas": k,
                   "acuracia_pct": round(100*k/n, 1), "ic95_lo": round(lo, 1), "ic95_hi": round(hi, 1)})

print("\n" + "=" * 100)
print("QUADRO RESUMO - NOTEBOOK 08 (sazonalidade e clima, regua de 4 marcos, versao final)")
print("=" * 100)
print(pd.DataFrame(resumo).to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Peso das variaveis (treinado com toda a base disponivel, horizonte t+1)
# ---------------------------------------------------------------------------
print("\n### PESO DAS VARIAVEIS (modelo treinado com toda a base, horizonte t+1) ###")
df1 = montar(1)
cols1 = [c for c in COLS_MODELO if c in df1.columns and df1[c].nunique(dropna=True) > 1]
m1 = modelo_final()
w1 = peso_amostra(df1["y"].values, df1["alvo_mes"].values)
m1.fit(df1[cols1], df1["y"], clf__sample_weight=w1)
coefs = pd.Series(m1.named_steps["clf"].coef_[0], index=cols1).sort_values(key=abs, ascending=False)
print(coefs.round(3).to_string())

# COMMAND ----------

sys.stdout = _old
_rel = _buf.getvalue()
print(_rel[-3000:])
dbutils.notebook.exit(_rel[-60000:])
