# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Regressao Logistica com Clima e EAR (regua de 4 marcos, modelo final)
# MAGIC Modelo final escolhido para prever a bandeira tarifaria vermelha
# MAGIC nos horizontes t+1, t+2 e t+3. Marcadores corrigidos para o mes de origem.
# MAGIC Demais componentes da receita preservados; resultados anteriores nao se transferem.
# MAGIC
# MAGIC Receita completa (fixada apos o teste comparativo entre os 6 notebooks de modelagem
# MAGIC ja existentes mais as variacoes testadas):
# MAGIC - Variaveis: bandeira do mes anterior (persistencia), mes do clima, chuva media, chuva
# MAGIC   percentual da normal (corrigida, ver item abaixo), temperatura, umidade, percentual da
# MAGIC   EAR (energia armazenada) do subsistema Sudeste.
# MAGIC - Regua de 4 marcos regulatorios (variaveis binarias que marcam mudancas reais de regra
# MAGIC   da ANEEL/ONS/CCEE: metodologia GSF x PLD horario dez/2018, revisao de faixas jun/2019,
# MAGIC   intervencao da pandemia mai-nov/2020, escassez hidrica set/2021-abr/2022). O marco mais
# MAGIC   recente, o novo abaco da REH 3.306/2024 (abr/2024), permanece fora da regua,
# MAGIC   conforme a especificacao escolhida. Sua omissao nao prova ganho de desempenho
# MAGIC   nem constitui, isoladamente, uma correcao de vazamento.
# MAGIC - Correcao temporal dos quatro marcos (11/09/2026): representam o regime observado
# MAGIC   no mes de origem fechado, NAO o regime realizado no mes-alvo futuro.
# MAGIC   Para a mesma origem, os quatro valores sao iguais em t+1, t+2 e t+3.
# MAGIC   Um inicio ou encerramento posterior a origem nao altera seus valores nessa origem.
# MAGIC   Nao se pressupoe conhecer o fim futuro de uma intervencao ainda ativa.
# MAGIC - Correcao do vazamento na normal de chuva: a normal usada para calcular o percentual de
# MAGIC   chuva e calculada de forma expansiva (media dos mesmos meses do calendario ocorridos
# MAGIC   antes de cada mes, nunca usando informacao futura).
# MAGIC - Peso 5x maior para observacoes a partir de abr/2024 (o regime mais parecido com o futuro).
# MAGIC - Reponderacao de classes (a bandeira vermelha e rara, sem isso o modelo tende a prever
# MAGIC   sempre nao-vermelha).
# MAGIC
# MAGIC Protocolo de teste: backtest longo por janela expansiva (rotulos de treino ate a origem),
# MAGIC treino minimo de 36 observacoes, alvos candidatos desde jan/2019 ate ago/2026.
# MAGIC O inicio efetivo depende do horizonte e do treino minimo; exclusoes sao registradas.
# MAGIC Avaliacao retrospectiva, nao um novo holdout intocado. A disponibilidade real dos
# MAGIC dados na data de divulgacao da bandeira exige verificacao propria.
# MAGIC Os marcos sao estados mensais historicos, nao uma base de comunicados com data de
# MAGIC publicacao auditada. A correcao retira o estado futuro do alvo, mas nao certifica
# MAGIC disponibilidade intramensal, bases sem revisoes ou antecedencia ao anuncio da ANEEL.
# MAGIC O peso recente e as demais escolhas continuam fixados retrospectivamente.
# MAGIC
# MAGIC Ajuste de integridade (11/09/2026): corte fixo, alvos oficiais independentes do clima,
# MAGIC validacao de calendario/chaves, leitura explicita da EAR e persistencia por horizonte.
# MAGIC Este notebook somente le as tabelas; nao altera dados nem outros notebooks.

# COMMAND ----------

import io, sys
_buf = io.StringIO(); _old = sys.stdout; sys.stdout = _buf

# COMMAND ----------

import math
import json
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
rng = np.random.default_rng(RANDOM_STATE)
CORTE_AAAAMM = 202608
FIM_DADOS = "2026-09-01"  # limite exclusivo
FIM_TESTE = pd.Period("2026-08", freq="M")
AUDITORIA = {"corte": CORTE_AAAAMM}
PREVISOES = []
EXCLUSOES = []


def para_periodo(aaaamm):
    aaaamm = int(aaaamm)
    return pd.Period(year=aaaamm // 100, month=aaaamm % 100, freq="M")


def validar_chaves(df, chave, nome):
    if df.empty:
        raise ValueError(f"{nome}: tabela vazia apos corte.")
    if df[chave].isna().any():
        raise ValueError(f"{nome}: chave nula.")
    if df[chave].duplicated().any():
        repetidas = df.loc[df[chave].duplicated(False), chave].tolist()
        raise ValueError(f"{nome}: chaves duplicadas: {repetidas[:20]}")


def validar_calendario(periodos, nome):
    idx = pd.PeriodIndex(periodos, freq="M").sort_values()
    faltantes = pd.period_range(idx.min(), idx.max(), freq="M").difference(idx)
    if len(faltantes):
        raise ValueError(f"{nome}: meses ausentes: {[str(x) for x in faltantes]}")
    return {"inicio": str(idx.min()), "fim": str(idx.max()), "meses": len(idx)}

# ---------------------------------------------------------------------------
# 1. Carga dos dados reais do Unity Catalog
# ---------------------------------------------------------------------------
print("[1] lendo mba.refined.f_modelo_bandeira_clima (clima INMET real)")
clima = spark.table("mba.refined.f_modelo_bandeira_clima").where(
    f"MesCompetencia <= {CORTE_AAAAMM} AND MesReferenciaClima < 202609"
).toPandas()
print("    linhas:", len(clima))

print("[2] lendo mba.trusted.f_bandeira (serie oficial do alvo)")
band = spark.table("mba.trusted.f_bandeira").where(
    f"MesCompetencia <= {CORTE_AAAAMM}"
).toPandas()
validar_chaves(clima, "MesCompetencia", "clima")
validar_chaves(clima, "MesReferenciaClima", "origens climaticas")
validar_chaves(band, "MesCompetencia", "bandeiras")
for col in ("MesCompetencia", "MesReferenciaClima"):
    clima[col] = pd.to_numeric(clima[col], errors="raise").astype(int)
band["MesCompetencia"] = pd.to_numeric(band["MesCompetencia"], errors="raise").astype(int)
if band["IsVermelha"].isna().any() or not band["IsVermelha"].isin([0, 1]).all():
    raise ValueError("A serie oficial deve conter apenas rotulos binarios 0/1.")
band["IsVermelha"] = band["IsVermelha"].astype(int)
AUDITORIA["bandeiras"] = validar_calendario(
    band["MesCompetencia"].map(para_periodo), "bandeiras"
)
AUDITORIA["clima"] = validar_calendario(
    clima["MesReferenciaClima"].map(para_periodo), "origens climaticas"
)
if band["MesCompetencia"].max() != CORTE_AAAAMM:
    raise ValueError("A serie oficial nao chega ao corte academico de agosto/2026.")
print("    linhas:", len(band), "| periodo:", band.MesCompetencia.min(), "a", band.MesCompetencia.max())

b = clima.sort_values("MesCompetencia").reset_index(drop=True).copy()
b = b.rename(columns={"MesReferenciaClima": "MesRef"})
serie_band = band.sort_values("MesCompetencia").set_index("MesCompetencia")["IsVermelha"]
serie_bandeira = pd.Series(
    serie_band.values, index=pd.PeriodIndex([para_periodo(x) for x in serie_band.index])
).sort_index()


def mes_menos(aaaamm, k):
    a, m = aaaamm // 100, aaaamm % 100
    t = a * 12 + (m - 1) - k
    return (t // 12) * 100 + (t % 12) + 1


if not all(para_periodo(ref) + 1 == para_periodo(alvo)
           for ref, alvo in zip(b["MesRef"], b["MesCompetencia"])):
    raise ValueError("MesReferenciaClima deve ser o mes anterior a MesCompetencia.")
b["IsVermelha"] = [serie_band.get(m, np.nan) for m in b.MesCompetencia]
b["IsVermelhaMesAnterior"] = [float(serie_band.get(m, np.nan)) for m in b.MesRef]
if b["IsVermelhaMesAnterior"].isna().any():
    raise ValueError("Falta bandeira oficial na origem: persistencia nao verificavel.")
# X depende da origem; y permanece na serie oficial completa, nunca recortado por X.
bruto = b.reset_index(drop=True)
print("\n[base montada] linhas:", len(bruto),
      "| periodo:", bruto.MesCompetencia.min(), "a", bruto.MesCompetencia.max(),
      "| vermelhas:", int(bruto.IsVermelha.sum()))


bruto["origem"] = bruto["MesRef"].map(para_periodo)
bruto["alvo_t1"] = bruto["MesCompetencia"].map(para_periodo)

FEATURES_ORIGEM = {
    "chuva_acum": "PrecipitacaoAcumuladaMm",
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
# 2. EAR do subsistema Sudeste (energia armazenada nos reservatorios)
# ---------------------------------------------------------------------------
print("\n[3] lendo mba.trusted.f_ear_subsistema (EAR, subsistema SE)")
_ear = spark.table("mba.trusted.f_ear_subsistema").select(
    F.to_date("ear_data").alias("Data"), "id_subsistema",
    F.col("ear_verif_subsistema_percentual").cast("double").alias("Valor"),
).filter(
    (F.col("id_subsistema") == "SE")
    & F.col("Data").isNotNull()
    & (F.col("Data") < F.to_date(F.lit(FIM_DADOS)))
).toPandas()
if _ear.empty:
    raise ValueError("EAR SE vazia apos filtro e corte.")
if _ear.duplicated(["Data", "id_subsistema"]).any():
    raise ValueError("EAR SE com datas duplicadas: revisar a origem antes de modelar.")
if _ear["Valor"].isna().all():
    raise ValueError("EAR SE sem nenhum valor numerico.")
_ear["mes"] = [pd.Period(year=x.year, month=x.month, freq="M") for x in _ear["Data"]]
s_ear = _ear.groupby("mes")["Valor"].mean()
print("    meses de EAR SE:", len(s_ear), "| de", s_ear.index.min(), "a", s_ear.index.max())
base["ear_pct"] = [float(s_ear.get(o, np.nan)) for o in base.index]
meses_sem_ear = [str(o) for o in base.index[base["ear_pct"].isna()]]
if meses_sem_ear:
    raise ValueError(f"EAR mensal ausente nas origens: {meses_sem_ear}")
AUDITORIA["ear"] = {
    "meses": len(s_ear), "inicio": str(s_ear.index.min()), "fim": str(s_ear.index.max()),
    "dias_com_valor_nulo": int(_ear["Valor"].isna().sum()),
    "cobertura_nota": "Media dos dias disponiveis; completude diaria ainda exige auditoria.",
}
if not np.isfinite(base[["chuva_acum", "chuva_media", "temperatura", "umidade", "ear_pct"]]
                   .astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()).all():
    raise ValueError("Variaveis climaticas/EAR com valores ausentes ou nao finitos.")
AUDITORIA["origens"] = validar_calendario(base.index, "base do modelo")

# ---------------------------------------------------------------------------
# 3. Correcao do vazamento na normal de chuva (so passado, nunca o futuro)
# ---------------------------------------------------------------------------
print("\n[4] corrigindo a normal de chuva (normal expansiva, so com o passado)")
ca = base["chuva_acum"].astype(float).values
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
base["chuva_pct_normal_ok"] = np.where(norm_exp > 0, ca / norm_exp * 100, np.nan)
idx_normal = np.flatnonzero(np.isfinite(norm_exp))
if not len(idx_normal):
    raise ValueError("Sem historico anterior para a normal expansiva.")
print("    primeiro mes com normal corrigida disponivel:", str(base.index[idx_normal[0]]))
AUDITORIA["normal_expansiva"] = {
    "primeira_origem": str(base.index[idx_normal[0]]),
    "origens_sem_normal_anterior": int(np.isnan(norm_exp).sum()),
    "tratamento": "Mediana ajustada somente no treino, preservando a receita.",
}

# ---------------------------------------------------------------------------
# 4. Regua de 4 marcos regulatorios OBSERVADOS NA ORIGEM, nunca no alvo
# ---------------------------------------------------------------------------
MARCOS_4 = {
    "regime_gsf_pld": (pd.Period("2018-12", "M"), None),
    "regime_faixas_2019": (pd.Period("2019-06", "M"), None),
    "intervencao_pandemia": (pd.Period("2020-05", "M"), pd.Period("2020-11", "M")),
    "intervencao_escassez": (pd.Period("2021-09", "M"), pd.Period("2022-04", "M")),
}


def marca_regime(meses, nome):
    """Estado historico no mes informado; em montar(), recebe somente origens fechadas."""
    ini, fim = MARCOS_4[nome]
    v = meses >= ini
    if fim is not None:
        v = v & (meses <= fim)
    return v.astype(int)


COLS_MODELO = ["mes_clima", "chuva_media", "chuva_pct_normal_ok", "temperatura", "umidade",
               "bandeira_origem", "ear_pct"] + list(MARCOS_4)


def montar(h):
    """Alvo = y(o+h); clima/EAR e estado regulatorio referem-se a origem fechada o.

    A disponibilidade intramensal/publicacao dos dados exige auditoria separada.
    """
    if h not in (1, 2, 3):
        raise ValueError("Horizonte deve ser 1, 2 ou 3 meses.")
    linhas = []
    for o, r in base.iterrows():
        alvo_mes = o + h
        if alvo_mes > FIM_TESTE or alvo_mes not in serie_bandeira.index:
            continue
        d = {"origem": o, "alvo_mes": alvo_mes, "y": int(serie_bandeira.loc[alvo_mes])}
        for c in ["mes_clima", "chuva_media", "chuva_pct_normal_ok", "temperatura",
                  "umidade", "bandeira_origem", "ear_pct"]:
            d[c] = r[c]
        linhas.append(d)
    df = pd.DataFrame(linhas)
    origens = pd.PeriodIndex(df["origem"], freq="M")
    for nome in MARCOS_4:
        df[nome] = marca_regime(origens, nome)
    auditar_marcos_origem(df, h)
    return df


def auditar_marcos_origem(df, h):
    """Falha explicitamente se qualquer marco representar o futuro em vez da origem."""
    divergencias = 0
    for nome, (inicio, fim) in MARCOS_4.items():
        esperado = np.array([
            int(o >= inicio and (fim is None or o <= fim)) for o in df["origem"]
        ])
        if not np.array_equal(df[nome].to_numpy(), esperado):
            raise ValueError(f"t+{h}: marcador {nome} divergente do estado na origem.")
        # Diagnostico: quantas celulas mudaram frente ao calendario ex post do alvo.
        antigo = marca_regime(pd.PeriodIndex(df["alvo_mes"], freq="M"), nome)
        divergencias += int(np.sum(esperado != antigo))
    AUDITORIA.setdefault("marcos_na_origem", {})[f"t+{h}"] = {
        "referencia": "origem mensal fechada; nao usa estado do mes-alvo",
        "linhas_validadas": len(df),
        "celulas_diferentes_do_calendario_no_alvo": divergencias,
        "publicacao_intramensal_auditada": False,
    }


# ---------------------------------------------------------------------------
# 5. Protocolo de teste: backtest longo, janela expansiva
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
    alvos_teste = [a for a in serie_bandeira.index if INICIO_TESTE <= a <= FIM_TESTE]
    cols = [c for c in COLS_MODELO if c in df.columns]
    yr, yp, meses = [], [], []
    for alvo in alvos_teste:
        lt = df[df["alvo_mes"] == alvo]
        if lt.empty:
            raise ValueError(f"t+{h}, alvo {alvo}: origem {alvo-h} sem variaveis.")
        if len(lt) != 1:
            raise ValueError(f"t+{h}, alvo {alvo}: mais de uma linha de origem.")
        o = lt["origem"].iloc[0]
        tr = df[df["alvo_mes"] <= o]
        if len(tr) < MIN_TREINO or tr["y"].nunique() < 2:
            EXCLUSOES.append({
                "horizonte": h, "alvo": str(alvo), "origem": str(o),
                "motivo": "treino minimo/classes", "n_treino": len(tr),
            })
            continue
        usadas = [c for c in cols if tr[c].nunique(dropna=True) > 1]
        if not usadas:
            raise ValueError(f"t+{h}, alvo {alvo}: todas as variaveis constantes.")
        m = modelo_final()
        w = peso_amostra(tr["y"].values, tr["alvo_mes"].values)
        try:
            m.fit(tr[usadas], tr["y"], clf__sample_weight=w)
            p = int(m.predict(lt[usadas])[0])
        except Exception as exc:
            raise RuntimeError(f"Falha em t+{h}, origem {o}, alvo {alvo}: {exc}") from exc
        PREVISOES.append({
            "horizonte": h, "origem": str(o), "alvo": str(alvo),
            "real": int(lt["y"].iloc[0]), "previsto": p,
            "persistencia": int(lt["bandeira_origem"].iloc[0]),
            "n_treino": len(tr), "ultima_data_alvo_treino": str(tr["alvo_mes"].max()),
            "mes_referencia_marcos": str(o),
            "marcos_origem": {nome: int(lt[nome].iloc[0]) for nome in MARCOS_4},
        })
        yr.append(int(lt["y"].iloc[0])); yp.append(p); meses.append(alvo)
    return np.array(yr), np.array(yp), meses


def mcnemar_exato(y, modelo, persistencia):
    """Diagnostico pareado; nao corrige dependencia temporal nem testes multiplos."""
    b = int(np.sum((modelo == y) & (persistencia != y)))
    c = int(np.sum((modelo != y) & (persistencia == y)))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c)+1)) / 2**n)
    return p, b, c


print("\n" + "#" * 108)
print("# BACKTEST LONGO: NOTEBOOK 07 (clima + EAR, 4 marcos NA ORIGEM, peso 5x pos abaco)")
print("#" * 108)
resumo = []
for h in (1, 2, 3):
    y, yp, meses = roda(h)
    k = int((y == yp).sum()); n = len(y)
    if n == 0:
        raise ValueError(f"t+{h}: nenhuma previsao avaliavel.")
    lo, hi = wilson(k, n)
    m = met(y, yp)
    persist = np.array([int(serie_bandeira.loc[a-h]) for a in meses])
    mp = met(y, persist)
    p_mc, vence_modelo, vence_persist = mcnemar_exato(y, yp, persist)
    print(f"\n--- t+{h} ---")
    print(f"  periodo de teste: {meses[0]} a {meses[-1]}  ({n} previsoes, {int(y.sum())} vermelhas)")
    print(f"  acuracia = {100*k/n:.1f}% ({k}/{n})  IC95% (Wilson) = [{lo:.1f}% ; {hi:.1f}%]")
    print(f"  F1 vermelha = {m['f1_vermelha']:.1f}%  precisao = {m['precisao']:.1f}%  recall = {m['recall']:.1f}%")
    print(f"  persistencia: acuracia = {mp['acuracia']:.1f}%  F1 vermelha = {mp['f1_vermelha']:.1f}%")
    print(f"  delta de acuracia = {m['acuracia']-mp['acuracia']:+.1f} p.p.; McNemar exploratorio p={p_mc:.4f}")
    resumo.append({"horizonte": f"t+{h}", "n": n, "corretas": k,
                   "inicio": str(meses[0]), "fim": str(meses[-1]),
                   "acuracia_pct": round(100*k/n, 1), "ic95_lo": round(lo, 1), "ic95_hi": round(hi, 1),
                   "f1_vermelha_pct": round(m["f1_vermelha"], 1),
                   "persistencia_acuracia_pct": round(mp["acuracia"], 1),
                   "persistencia_f1_vermelha_pct": round(mp["f1_vermelha"], 1),
                   "delta_acuracia_pp": round(m["acuracia"]-mp["acuracia"], 1),
                   "mcnemar_p": p_mc, "discordantes_modelo_acerta": vence_modelo,
                   "discordantes_persistencia_acerta": vence_persist})

print("\n" + "=" * 100)
print("QUADRO RESUMO - NOTEBOOK 07 (clima + EAR, 4 marcos NA ORIGEM)")
print("=" * 100)
print(pd.DataFrame(resumo).to_string(index=False))
print("\nEXCLUSOES EXPLICITAS (antes de atingir o treino minimo):")
print(pd.DataFrame(EXCLUSOES).to_string(index=False))
print("\nPREVISOES POR ORIGEM E ALVO:")
print(pd.DataFrame(PREVISOES).to_string(index=False))
print("\nVALIDACOES:")
print(json.dumps(AUDITORIA, ensure_ascii=False, indent=2))
print("\nNota: avaliacao retrospectiva; Wilson/McNemar nao controlam dependencia temporal.")
print("Marcos representam a origem fechada, nao o alvo. Publicacao intramensal nao auditada.")

# ---------------------------------------------------------------------------
# 6. Peso das variaveis (treinado com toda a base disponivel, horizonte t+1)
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

# MAGIC %md
# MAGIC ## 7. Previsao real (fora da amostra) para os meses ainda sem bandeira publicada
# MAGIC O backtest acima (secao 5) so avalia meses de alvo que ja tem a bandeira oficial
# MAGIC publicada em `mba.trusted.f_bandeira` -- por isso ele nunca "avanca" para alem do
# MAGIC ultimo mes com rotulo conhecido. Esta secao treina os mesmos modelos (mesma
# MAGIC receita de features e pesos) com TODO o historico disponivel e projeta os meses
# MAGIC de alvo que ainda NAO tem bandeira publicada -- a previsao real de T3.

# COMMAND ----------

print("\n" + "#" * 108)
print("# PREVISAO REAL (fora da amostra) - meses de alvo SEM bandeira oficial publicada")
print("#" * 108)


def construir_linha_origem(o):
    """Monta a linha de features do mes de origem `o` para gerar previsao.
    Se `o` ja existir em `base` (clima INMET completo ja processado), usa os
    valores reais. Caso `o` seja mais recente que o clima do INMET (ainda nao
    chegou na tabela refined), monta uma linha parcial só com o que já
    fechou: mes calendario, EAR do subsistema SE (que atualiza mais rapido) e
    a bandeira do proprio mes de origem (persistencia, ja publicada). As
    variaveis de clima ficam como NaN e sao preenchidas pela mediana do
    treino pelo SimpleImputer do pipeline.
    """
    if o in base.index:
        return base.loc[[o]].copy()
    linha = {c: np.nan for c in base.columns}
    linha["mes_clima"] = o.month
    linha["ear_pct"] = float(s_ear.get(o, np.nan))
    aaaamm_o = o.year * 100 + o.month
    linha["bandeira_origem"] = float(serie_band.get(aaaamm_o, np.nan))
    return pd.DataFrame([linha], index=[o])


# Origem usada para prever = ultimo mes JA FECHADO (bandeira ja conhecida),
# nao o ultimo mes com clima INMET completo -- assim t+1 mira o proximo mes
# de verdade (ex.: hoje ago/2026 fechado -> t+1 preve set/2026), em vez de
# repetir um mes que ja e conhecido.
origem_previsao = para_periodo(int(serie_band.index.max()))
tem_clima_completo = origem_previsao in base.index
print(f"mes de origem usado para prever (ultimo mes ja fechado)  : {origem_previsao}")
print(f"clima INMET completo disponivel para esse mes de origem  : "
      f"{'sim' if tem_clima_completo else 'nao -- usando EAR + persistencia; clima imputado pela mediana do treino'}")

x_row_base = construir_linha_origem(origem_previsao)

previsoes_futuras = []
for h in (1, 2, 3):
    alvo_h = origem_previsao + h
    ja_conhecido = alvo_h in serie_bandeira.index
    df_h = montar(h)
    cols_h = [c for c in COLS_MODELO if c in df_h.columns and df_h[c].nunique(dropna=True) > 1]
    if len(df_h) < MIN_TREINO:
        print(f"\nt+{h}: historico insuficiente para treinar ({len(df_h)} linhas)")
        continue

    m_h = modelo_final()
    w_h = peso_amostra(df_h["y"].values, df_h["alvo_mes"].values)
    m_h.fit(df_h[cols_h], df_h["y"], clf__sample_weight=w_h)

    x_row = x_row_base.copy()
    alvos_fake = pd.PeriodIndex([alvo_h])
    for nome in MARCOS_4:
        x_row[nome] = marca_regime(alvos_fake, nome)[0]
    x_row = x_row[cols_h]

    proba = float(m_h.predict_proba(x_row)[0, 1])
    pred = int(proba >= 0.5)
    status = "JA PUBLICADA (nao e previsao real)" if ja_conhecido else "PREVISAO REAL (ainda nao publicada)"
    print(f"\nt+{h}: origem={origem_previsao} -> alvo={alvo_h}  [{status}]")
    print(f"  treinado com {len(df_h)} meses historicos ({int(df_h['y'].sum())} vermelhas)")
    print(f"  P(vermelha) = {proba*100:.1f}%   classe prevista = {'VERMELHA' if pred else 'nao-vermelha'}")
    previsoes_futuras.append({
        "horizonte": f"t+{h}", "origem": str(origem_previsao), "alvo_mes": str(alvo_h),
        "status": status, "n_treino": len(df_h),
        "prob_vermelha_pct": round(proba * 100, 1),
        "classe_prevista": "VERMELHA" if pred else "nao-vermelha",
    })

print("\n" + "=" * 100)
print("QUADRO - PREVISAO REAL PARA OS PROXIMOS MESES (a partir do ultimo mes fechado)")
print("=" * 100)
print(pd.DataFrame(previsoes_futuras).to_string(index=False))

# COMMAND ----------

sys.stdout = _old
_rel = _buf.getvalue()
print(_rel)
dbutils.notebook.exit(json.dumps({
    "notebook": "07_regressao_logistica_clima_ear_regua_final",
    "versao_marcos": "origem_mensal_fechada_20260911",
    "validacoes": AUDITORIA, "resumo": resumo, "exclusoes": EXCLUSOES,
    "previsoes": PREVISOES, "coeficientes_t1": {str(k): float(v) for k, v in coefs.items()},
    "relatorio": _rel,
}, ensure_ascii=False))
