# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Regressao Logistica (modelo final)
# MAGIC Modelo escolhido para prever a bandeira tarifaria vermelha nos horizontes t+1, t+2 e t+3.
# MAGIC
# MAGIC Receita fixada apos testes (notebooks internos de validacao, ja descartados):
# MAGIC - Base completa desde fevereiro de 2016 (nao corta o periodo de pandemia/crise hidrica,
# MAGIC   trata esses periodos como variaveis de regime em vez de removê-los).
# MAGIC - Variaveis: bandeira do mes anterior (persistencia), sazonalidade (seno/cosseno do mes-alvo),
# MAGIC   chuva acumulada, chuva % da normal, chuva % da normal em 3 meses, temperatura, umidade.
# MAGIC - Cinco variaveis de regime regulatorio (marcam mudancas de regra da ANEEL/ONS: metodologia
# MAGIC   GSF x PLD dez/2018, revisao de faixas jun/2019, pandemia mai-nov/2020, escassez hidrica
# MAGIC   set/2021-abr/2022, novo abaco da REH 3.306/2024 a partir de abr/2024).
# MAGIC - Peso 5x maior para observacoes a partir de abr/2024 (o regime mais parecido com o futuro).
# MAGIC - Reponderacao de classes (a bandeira vermelha e rara, sem isso o modelo "chuta" sempre nao-vermelha).
# MAGIC
# MAGIC Janela de teste: abril/2025 a agosto/2026 (17 meses-alvo, 6 com bandeira vermelha real).
# MAGIC Validacao por janela expansiva (o modelo so ve o passado de cada mes, nunca o futuro).

# COMMAND ----------

import io, sys
_buf = io.StringIO(); _old = sys.stdout; sys.stdout = _buf

# COMMAND ----------

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
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

b = b.set_index("MesRef")
b["PrecipitacaoAcumulada3M"] = b["PrecipitacaoAcumuladaMm"].rolling(3, min_periods=1).sum()
b["PrecipitacaoNormal3M"] = b["PrecipitacaoNormalMm"].rolling(3, min_periods=1).mean()
b["PrecipitacaoPctNormal3M"] = np.where(
    b["PrecipitacaoNormal3M"] > 0, b["PrecipitacaoAcumulada3M"] / b["PrecipitacaoNormal3M"] * 100, np.nan)
b = b.reset_index()

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
    "chuva_pct_normal": "PrecipitacaoPctNormal",
    "chuva_pct_normal_3m": "PrecipitacaoPctNormal3M",
    "temperatura": "TemperaturaMediaC",
    "umidade": "UmidadeMediaPct",
    "bandeira_origem": "IsVermelhaMesAnterior",
}
base = pd.DataFrame({"origem": bruto["origem"]})
for novo, antigo in FEATURES_ORIGEM.items():
    base[novo] = bruto[antigo].values
base = base.set_index("origem").sort_index()

# ---------------------------------------------------------------------------
# 2. Dummies de regime regulatorio (a "regua" que mudou ao longo do tempo)
# ---------------------------------------------------------------------------
MARCOS = {
    "regime_gsf_pld": (pd.Period("2018-12", "M"), None),
    "regime_faixas_2019": (pd.Period("2019-06", "M"), None),
    "intervencao_pandemia": (pd.Period("2020-05", "M"), pd.Period("2020-11", "M")),
    "intervencao_escassez": (pd.Period("2021-09", "M"), pd.Period("2022-04", "M")),
    "regime_abaco_2024": (pd.Period("2024-04", "M"), None),
}


def marca_regime(meses, nome):
    ini, fim = MARCOS[nome]
    v = meses >= ini
    if fim is not None:
        v = v & (meses <= fim)
    return v.astype(int)


def montar(h):
    """Uma linha por mes de origem o. Alvo = y(o+h). So usa informacao conhecida em o."""
    linhas = []
    for o, r in base.iterrows():
        alvo_mes = o + h
        if alvo_mes not in serie_bandeira.index:
            continue
        d = {"origem": o, "alvo_mes": alvo_mes, "y": int(serie_bandeira.loc[alvo_mes])}
        d.update({k: r[k] for k in FEATURES_ORIGEM})
        m = alvo_mes.month
        d["sin_mes"] = np.sin(2 * np.pi * m / 12)
        d["cos_mes"] = np.cos(2 * np.pi * m / 12)
        linhas.append(d)
    df = pd.DataFrame(linhas)
    alvos = pd.PeriodIndex(df["alvo_mes"])
    for nome in MARCOS:
        df[nome] = marca_regime(alvos, nome)
    return df


COLS_H4 = ["bandeira_origem", "sin_mes", "cos_mes", "chuva_acum", "chuva_pct_normal",
           "chuva_pct_normal_3m", "temperatura", "umidade"] + list(MARCOS)

# ---------------------------------------------------------------------------
# 3. Protocolo de teste: janela expansiva, mesma janela para os 3 horizontes
# ---------------------------------------------------------------------------
INICIO_TESTE = pd.Period("2025-04", "M")
POS_ABACO = pd.Period("2024-04", "M")
MIN_TREINO = 10
PESO_RECENCIA = 5.0

todos_alvos = sorted(serie_bandeira.index)
ALVOS = [a for a in todos_alvos if a >= INICIO_TESTE]

print("=" * 100)
print("MODELO FINAL: Regressao Logistica, receita H4")
print(f"Janela de teste: {ALVOS[0]} a {ALVOS[-1]}  "
      f"({len(ALVOS)} meses-alvo | {int(serie_bandeira.loc[ALVOS].sum())} vermelhas)")
print("=" * 100)


def met(y, yp, pr=None):
    d = dict(n=len(y), n_verm_real=int(y.sum()), n_verm_prev=int(yp.sum()),
             acuracia=accuracy_score(y, yp) * 100,
             precisao=precision_score(y, yp, zero_division=0) * 100,
             recall=recall_score(y, yp, zero_division=0) * 100,
             f1_vermelha=f1_score(y, yp, zero_division=0) * 100,
             f1_macro=f1_score(y, yp, average="macro", zero_division=0) * 100)
    d["roc_auc"] = roc_auc_score(y, pr) * 100 if (pr is not None and len(set(y)) > 1) else float("nan")
    return d


def ic_acc(y, yp, n=5000):
    v = []; idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(y), replace=True)
        v.append(accuracy_score(y[s], yp[s]) * 100)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def mcnemar(y, a, b):
    n01 = int(((a == y) & (b != y)).sum()); n10 = int(((b == y) & (a != y)).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def peso_amostra(y_tr, alvos_tr):
    classes, counts = np.unique(y_tr, return_counts=True)
    n = len(y_tr)
    cw = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    w_classe = np.array([cw[v] for v in y_tr])
    w_recencia = np.where(pd.PeriodIndex(alvos_tr) >= POS_ABACO, PESO_RECENCIA, 1.0)
    return w_classe * w_recencia


def modelo_final():
    clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("clf", clf)])


def roda(h, min_treino=MIN_TREINO):
    df = montar(h)
    cols = [c for c in COLS_H4 if c in df.columns]
    linhas_previsao = []
    yr, yp, pb = [], [], []
    for alvo in ALVOS:
        lt = df[df["alvo_mes"] == alvo]
        if lt.empty:
            return None
        o = lt["origem"].iloc[0]
        tr = df[df["alvo_mes"] <= o]
        if len(tr) < min_treino or tr["y"].nunique() < 2:
            return None
        usadas = [c for c in cols if tr[c].nunique() > 1]
        m = modelo_final()
        w = peso_amostra(tr["y"].values, tr["alvo_mes"].values)
        m.fit(tr[usadas], tr["y"], clf__sample_weight=w)
        p = float(m.predict_proba(lt[usadas])[0, 1])
        real = int(lt["y"].iloc[0]); prev = int(p >= 0.5)
        yr.append(real); pb.append(p); yp.append(prev)
        linhas_previsao.append({"mes_alvo": str(alvo), "real": real, "previsto": prev,
                                 "probabilidade_vermelha_pct": round(p * 100, 1)})
    return np.array(yr), np.array(yp), np.array(pb), linhas_previsao


print("\n### RESULTADOS POR HORIZONTE ###")
resumo = []
tabelas_mensais = {}
for h in (1, 2, 3):
    out = roda(h)
    y, yp, pb, linhas = out
    m = met(y, yp, pb)
    lo, hi = ic_acc(y, yp)
    persist = np.array([1 if v >= 0.5 else 0 for v in
                         montar(h).set_index("alvo_mes").loc[ALVOS, "bandeira_origem"]])
    p_mcnemar = mcnemar(y, yp, persist)
    print(f"\n--- t+{h} ---")
    print(f"  acuracia = {m['acuracia']:.1f}%  IC95%[{lo:.1f};{hi:.1f}]  "
          f"F1 vermelha = {m['f1_vermelha']:.1f}%  F1 macro = {m['f1_macro']:.1f}%  AUC = {m['roc_auc']:.1f}%")
    print(f"  vermelhas reais = {m['n_verm_real']}  previstas = {m['n_verm_prev']}  "
          f"precisao = {m['precisao']:.1f}%  recall = {m['recall']:.1f}%")
    print(f"  McNemar (vs persistencia simples) p = {p_mcnemar:.4f}")
    print(f"  {'mes-alvo':<10} {'real':<6} {'previsto':<10} {'prob. vermelha'}")
    for ln in linhas:
        print(f"  {ln['mes_alvo']:<10} {ln['real']:<6} {ln['previsto']:<10} {ln['probabilidade_vermelha_pct']}%")
    resumo.append({"horizonte": f"t+{h}", "acuracia_pct": round(m["acuracia"], 1),
                   "f1_vermelha_pct": round(m["f1_vermelha"], 1), "f1_macro_pct": round(m["f1_macro"], 1),
                   "auc_pct": round(m["roc_auc"], 1), "mcnemar_p": round(p_mcnemar, 4)})
    tabelas_mensais[h] = linhas

print("\n" + "=" * 100)
print("QUADRO RESUMO - modelo final (Regressao Logistica, receita H4)")
print("=" * 100)
print(pd.DataFrame(resumo).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Peso das variaveis (treinado com toda a base disponivel, para leitura)
# ---------------------------------------------------------------------------
print("\n### PESO DAS VARIAVEIS (modelo treinado com toda a base, horizonte t+1) ###")
df1 = montar(1)
cols1 = [c for c in COLS_H4 if c in df1.columns and df1[c].nunique() > 1]
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
