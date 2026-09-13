# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Previsao da bandeira para um mes especifico (parametrizado)
# MAGIC Usa a mesma receita e o mesmo protocolo do notebook 07 (clima + EAR, regua de 4 marcos,
# MAGIC peso 5x pos abaco), mas em vez de rodar o backtest inteiro, recebe um mes alvo por
# MAGIC parametro (widget) e devolve so a previsao daquele mes.
# MAGIC
# MAGIC Regra de seguranca: o modelo nunca usa o clima do proprio mes que esta prevendo. Ele usa
# MAGIC sempre o clima do mes de origem mais recente disponivel (hoje, agosto/2026, ja completo) e
# MAGIC calcula automaticamente quantos meses de distancia (horizonte t+1, t+2 ou t+3) isso
# MAGIC representa até o mes alvo pedido. So aceita horizontes de 1 a 3 meses, que sao os unicos
# MAGIC testados e validados no notebook 07.
# MAGIC
# MAGIC Esta previsao NAO e gravada em nenhuma tabela (nao altera a refined). E so uma consulta.

# COMMAND ----------

dbutils.widgets.text("mes_alvo", "202611", "Mes alvo a prever (AAAAMM)")
MES_ALVO = int(dbutils.widgets.get("mes_alvo").strip())

# COMMAND ----------

import io, sys
_buf = io.StringIO(); _old = sys.stdout; sys.stdout = _buf

# COMMAND ----------

import math
import warnings
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Carga dos dados reais do Unity Catalog (identico ao notebook 07)
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
    "chuva_pct_normal": "PrecipitacaoPctNormal",
    "temperatura": "TemperaturaMediaC",
    "umidade": "UmidadeMediaPct",
    "bandeira_origem": "IsVermelhaMesAnterior",
    "chuva_media": "PrecipitacaoMediaMm",
}
base = pd.DataFrame({"origem": bruto["origem"]})
for novo, antigo in FEATURES_ORIGEM.items():
    base[novo] = bruto[antigo].values
base["mes_clima"] = [o.month for o in base["origem"]]
base = base.set_index("origem").sort_index()

# ---------------------------------------------------------------------------
# 2. EAR do subsistema Sudeste
# ---------------------------------------------------------------------------
print("\n[3] lendo mba.trusted.f_ear_subsistema (EAR, subsistema SE)")
_ear = spark.table("mba.trusted.f_ear_subsistema").select(
    F.to_date("ear_data").alias("Data"), "id_subsistema",
    F.col("ear_verif_subsistema_percentual").cast("double").alias("Valor"),
).filter((F.col("id_subsistema") == "SE") & F.col("ear_data").isNotNull()).toPandas()
_ear["mes"] = [pd.Period(year=x.year, month=x.month, freq="M") for x in _ear["Data"]]
s_ear = _ear.groupby("mes")["Valor"].mean()
print("    meses de EAR SE:", len(s_ear), "| de", s_ear.index.min(), "a", s_ear.index.max())
base["ear_pct"] = [float(s_ear.get(o, np.nan)) for o in base.index]

# ---------------------------------------------------------------------------
# 3. Correcao do vazamento na normal de chuva (so passado, nunca o futuro)
# ---------------------------------------------------------------------------
print("\n[4] corrigindo a normal de chuva (normal expansiva, so com o passado)")
ca = base["chuva_acum"].astype(float).values
mes_cal = np.array([p.month for p in base.index])
n_base = len(base)


def normal_expansiva(serie):
    out = np.full(n_base, np.nan)
    for i in range(n_base):
        ant = [serie[j] for j in range(i) if mes_cal[j] == mes_cal[i] and not np.isnan(serie[j])]
        if ant:
            out[i] = float(np.mean(ant))
    return out


norm_exp = normal_expansiva(ca)
base["chuva_pct_normal_ok"] = np.where(norm_exp > 0, ca / norm_exp * 100, np.nan)

# ---------------------------------------------------------------------------
# 4. Regua de 4 marcos regulatorios (identico ao notebook 07)
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


COLS_MODELO = ["mes_clima", "chuva_media", "chuva_pct_normal_ok", "temperatura", "umidade",
               "bandeira_origem", "ear_pct"] + list(MARCOS_4)


def montar(h):
    """Uma linha por mes de origem o. Alvo = y(o+h). So usa informacao conhecida em o."""
    linhas = []
    for o, r in base.iterrows():
        alvo_mes = o + h
        if alvo_mes not in serie_bandeira.index:
            continue
        d = {"origem": o, "alvo_mes": alvo_mes, "y": int(serie_bandeira.loc[alvo_mes])}
        for c in ["mes_clima", "chuva_media", "chuva_pct_normal_ok", "temperatura",
                  "umidade", "bandeira_origem", "ear_pct"]:
            d[c] = r[c]
        linhas.append(d)
    df = pd.DataFrame(linhas)
    alvos = pd.PeriodIndex(df["alvo_mes"])
    for nome in MARCOS_4:
        df[nome] = marca_regime(alvos, nome)
    return df


MIN_TREINO = 36
PESO_RECENTE = 5.0
POS_ABACO = pd.Period("2024-04", "M")


def modelo_final():
    clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("clf", clf)])


def peso_amostra(y_tr, alvos_tr):
    classes, counts = np.unique(y_tr, return_counts=True)
    n = len(y_tr)
    cw = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    w_classe = np.array([cw[v] for v in y_tr])
    w_recente = np.where(pd.PeriodIndex(alvos_tr) >= POS_ABACO, PESO_RECENTE, 1.0)
    return w_classe * w_recente


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - e), 100 * min(1, c + e))


# ---------------------------------------------------------------------------
# 5. Descobre automaticamente o horizonte (t+1, t+2 ou t+3) para o mes alvo
# ---------------------------------------------------------------------------
alvo_periodo = para_periodo(MES_ALVO)
origem_mais_recente = base.index.max()
h = (alvo_periodo.year * 12 + alvo_periodo.month) - (origem_mais_recente.year * 12 + origem_mais_recente.month)

print("\n" + "#" * 108)
print(f"# PREVISAO PARA {alvo_periodo}  (mes alvo pedido = {MES_ALVO})")
print("#" * 108)
print(f"  origem de clima mais recente disponivel: {origem_mais_recente} (ja completo)")
print(f"  horizonte necessario: t+{h}")

if h < 1 or h > 3:
    print(f"\n  [PARADO] O horizonte t+{h} esta fora da faixa testada e validada (t+1 a t+3).")
    if h < 1:
        print("  O mes pedido e igual ou anterior ao mes de origem mais recente; "
              "ele ja deveria ter resposta oficial conhecida, confira mba.trusted.f_bandeira.")
    else:
        print("  Para prever esse mes seria preciso esperar mais dados de clima chegarem "
              "(mais um mes de origem completo), ou usar um horizonte maior, o que nao foi validado.")
    resultado_txt = f"PARADO: horizonte t+{h} fora da faixa validada (1 a 3)."
else:
    ja_conhecido = alvo_periodo in serie_bandeira.index
    if ja_conhecido:
        print(f"\n  [AVISO] Este mes JA TEM bandeira oficial anunciada pela ANEEL: "
              f"{'VERMELHA' if serie_bandeira.loc[alvo_periodo] == 1 else 'NAO VERMELHA'}. "
              f"A previsao abaixo e so para comparacao, nao e mais uma previsao real do futuro.")

    df_h = montar(h)
    cols = [c for c in COLS_MODELO if c in df_h.columns]

    # treino: so com meses cujo alvo ja e conhecido ATE o mes de origem mais recente (sem vazamento)
    tr = df_h[df_h["alvo_mes"] <= origem_mais_recente]
    usadas = [c for c in cols if tr[c].nunique(dropna=True) > 1]

    print(f"\n  treino: {len(tr)} meses (de {tr['alvo_mes'].min()} a {tr['alvo_mes'].max()}), "
          f"{int(tr['y'].sum())} vermelhas | variaveis usadas: {len(usadas)}")

    if len(tr) < MIN_TREINO or tr["y"].nunique() < 2:
        print("\n  [PARADO] Nao ha treino suficiente ou so uma classe presente.")
        resultado_txt = "PARADO: treino insuficiente."
    else:
        m = modelo_final()
        w = peso_amostra(tr["y"].values, tr["alvo_mes"].values)
        m.fit(tr[usadas], tr["y"], clf__sample_weight=w)

        linha_pred = base.loc[[origem_mais_recente]].copy()
        for nome in MARCOS_4:
            linha_pred[nome] = marca_regime(pd.PeriodIndex([alvo_periodo]), nome)
        pred_classe = int(m.predict(linha_pred[usadas])[0])
        pred_proba = float(m.predict_proba(linha_pred[usadas])[0, 1])

        # acuracia historica desse horizonte (backtest, mesmo protocolo do notebook 07),
        # para contextualizar a confianca da previsao
        alvos_teste = [a for a in sorted(df_h["alvo_mes"].unique()) if a >= pd.Period("2019-01", "M")]
        yr, yp = [], []
        for alvo in alvos_teste:
            lt = df_h[df_h["alvo_mes"] == alvo]
            o = lt["origem"].iloc[0]
            tr_bt = df_h[df_h["alvo_mes"] <= o]
            if len(tr_bt) < MIN_TREINO or tr_bt["y"].nunique() < 2:
                continue
            usadas_bt = [c for c in cols if tr_bt[c].nunique(dropna=True) > 1]
            if not usadas_bt:
                continue
            m_bt = modelo_final()
            w_bt = peso_amostra(tr_bt["y"].values, tr_bt["alvo_mes"].values)
            try:
                m_bt.fit(tr_bt[usadas_bt], tr_bt["y"], clf__sample_weight=w_bt)
                p_bt = int(m_bt.predict(lt[usadas_bt])[0])
            except Exception:
                continue
            yr.append(int(lt["y"].iloc[0])); yp.append(p_bt)
        yr, yp = np.array(yr), np.array(yp)
        k, n = int((yr == yp).sum()), len(yr)
        lo, hi = wilson(k, n)

        print("\n" + "=" * 100)
        print(f"RESULTADO - PREVISAO PARA {alvo_periodo} (horizonte t+{h}, a partir do clima de {origem_mais_recente})")
        print("=" * 100)
        print(f"  Previsao do modelo: {'VERMELHA' if pred_classe == 1 else 'NAO VERMELHA'}")
        print(f"  Probabilidade estimada de vermelha: {pred_proba*100:.1f}%")
        print(f"  Acuracia historica do horizonte t+{h} (backtest longo, mesmo protocolo do notebook 07): "
              f"{100*k/n:.1f}% ({k}/{n})  IC95% (Wilson) = [{lo:.1f}% ; {hi:.1f}%]")
        print(f"  Esta previsao NAO foi gravada em nenhuma tabela. E so uma consulta pontual.")

        resultado_txt = (f"MES_ALVO={alvo_periodo} | HORIZONTE=t+{h} | ORIGEM_CLIMA={origem_mais_recente} | "
                         f"PREVISAO={'VERMELHA' if pred_classe == 1 else 'NAO_VERMELHA'} | "
                         f"PROBABILIDADE_VERMELHA={pred_proba*100:.1f}% | "
                         f"ACURACIA_HISTORICA_HORIZONTE={100*k/n:.1f}% ({k}/{n}) | "
                         f"IC95=[{lo:.1f}%;{hi:.1f}%]")

# COMMAND ----------

sys.stdout = _old
_rel = _buf.getvalue()
print(_rel[-4000:])
dbutils.notebook.exit(_rel[-60000:])
