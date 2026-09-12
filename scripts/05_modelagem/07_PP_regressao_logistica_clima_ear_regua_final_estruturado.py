# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Regressao Logistica com Clima e EAR (regua de 4 marcos, modelo final)
# MAGIC Modelo escolhido (junto do notebook 08) para prever a bandeira tarifaria vermelha
# MAGIC no horizonte t+1, com o protocolo unificado de teste.
# MAGIC
# MAGIC Receita completa (fixada apos o teste comparativo entre os 6 notebooks de modelagem
# MAGIC ja existentes mais as variacoes testadas):
# MAGIC - Variaveis: bandeira do mes anterior (persistencia), mes do clima, chuva media, chuva
# MAGIC   percentual da normal (corrigida, ver item abaixo), temperatura, umidade, percentual da
# MAGIC   EAR (energia armazenada) do subsistema Sudeste.
# MAGIC - Regua de 4 marcos regulatorios (variaveis binarias que marcam mudancas reais de regra
# MAGIC   da ANEEL/ONS/CCEE: metodologia GSF x PLD horario dez/2018, revisao de faixas jun/2019,
# MAGIC   intervencao da pandemia mai-nov/2020, escassez hidrica set/2021-abr/2022). O marco mais
# MAGIC   recente, o novo abaco da REH 3.306/2024 (abr/2024), fica de fora da regua de proposito,
# MAGIC   para nao contaminar o teste com uma variavel muito perto do periodo avaliado.
# MAGIC - Correcao do vazamento na normal de chuva: a normal usada para calcular o percentual de
# MAGIC   chuva e calculada de forma expansiva (media dos mesmos meses do calendario ocorridos
# MAGIC   antes de cada mes, nunca usando informacao futura).
# MAGIC - Peso 5x maior para observacoes a partir de abr/2024 (o regime mais parecido com o futuro).
# MAGIC - Na inferencia, as seis features de origem sao recebidas por widgets/parametros
# MAGIC   (`chuva_acum`, `chuva_pct_normal`, `temperatura`, `umidade`, `bandeira_origem` e
# MAGIC   `chuva_media`) e substituem os valores historicos apenas na linha consultada.
# MAGIC - Reponderacao de classes (a bandeira vermelha e rara, sem isso o modelo tende a prever
# MAGIC   sempre nao-vermelha).
# MAGIC
# MAGIC Protocolo de teste: backtest longo por janela expansiva (o modelo so ve o passado de cada
# MAGIC mes, nunca o futuro), treino minimo de 36 meses, teste a partir de jan/2019 (t+1), abr/2019
# MAGIC (t+2) ou jun/2019 (t+3), ate ago/2026.

# COMMAND ----------

import io, sys
_buf = io.StringIO()
_old = sys.stdout
sys.stdout = _buf


# COMMAND ----------

import math

widgets = dbutils.widgets.getAll()

# As chaves são os nomes lógicos usados pelo modelo; os valores são as colunas
# correspondentes na base histórica.
FEATURES_ORIGEM = {
    "chuva_acum": "PrecipitacaoAcumuladaMm",
    "chuva_pct_normal": "PrecipitacaoPctNormal",
    "temperatura": "TemperaturaMediaC",
    "umidade": "UmidadeMediaPct",
    "bandeira_origem": "IsVermelhaMesAnterior",
    "chuva_media": "PrecipitacaoMediaMm",
}

LABELS_FEATURES_ORIGEM = {
    "chuva_acum": "Precipitação acumulada (mm)",
    "chuva_pct_normal": "Precipitação percentual da normal (%)",
    "temperatura": "Temperatura média (°C)",
    "umidade": "Umidade média (%)",
    "bandeira_origem": "Bandeira vermelha no mês anterior (0/1)",
    "chuva_media": "Precipitação média (mm)",
}

# Cria somente os parâmetros expostos como widgets. bandeira_origem permanece
# no código e no conjunto de features do modelo, mas é fixada internamente em 0.
PARAMETROS_VISIVEIS = [
    "chuva_acum", "chuva_pct_normal", "temperatura", "umidade", "chuva_media",
]
for nome in PARAMETROS_VISIVEIS:
    descricao = LABELS_FEATURES_ORIGEM[nome]
    if nome not in widgets:
        dbutils.widgets.text(nome, "", descricao)

def _ler_float(nome):
    valor = dbutils.widgets.get(nome).strip()
    if not valor:
        raise ValueError(
            f"O parâmetro '{nome}' é obrigatório para a previsão parametrizada."
        )
    try:
        numero = float(valor.replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"O parâmetro '{nome}' deve ser numérico: {valor!r}") from exc
    if not math.isfinite(numero):
        raise ValueError(f"O parâmetro '{nome}' deve ser finito: {valor!r}")
    return numero


def _ler_bandeira(nome):
    valor = dbutils.widgets.get(nome).strip()
    if valor not in {"0", "1"}:
        raise ValueError(f"O parâmetro '{nome}' deve ser 0 ou 1: {valor!r}")
    return int(valor)


parametros_origem = {
    "chuva_acum": _ler_float("chuva_acum"),
    "chuva_pct_normal": _ler_float("chuva_pct_normal"),
    "temperatura": _ler_float("temperatura"),
    "umidade": _ler_float("umidade"),
    # Mantida como feature do modelo, mas sem entrada externa.
    "bandeira_origem": 0,
    "chuva_media": _ler_float("chuva_media"),
}

if parametros_origem["chuva_acum"] < 0:
    raise ValueError("O parâmetro 'chuva_acum' não pode ser negativo")
if parametros_origem["chuva_pct_normal"] < 0:
    raise ValueError("O parâmetro 'chuva_pct_normal' não pode ser negativo")
if parametros_origem["chuva_media"] < 0:
    raise ValueError("O parâmetro 'chuva_media' não pode ser negativo")
if not 0 <= parametros_origem["umidade"] <= 100:
    raise ValueError("O parâmetro 'umidade' deve estar entre 0 e 100")

print("Features de origem recebidas:", parametros_origem)


# COMMAND ----------

import math
import warnings
import json
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

# O portal não envia ano ou mês. O período de referência será escolhido
# automaticamente depois que a base climática for carregada.

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
    aaaamm = int(aaaamm)
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

base = pd.DataFrame({"origem": bruto["origem"]})
for novo, antigo in FEATURES_ORIGEM.items():
    base[novo] = bruto[antigo].values
base["mes_clima"] = [o.month for o in base["origem"]]
base = base.set_index("origem").sort_index()
if base.empty:
    raise ValueError("A base climática não possui linhas disponíveis para previsão.")

# Sem entrada de mês/ano no portal, usa o último período de origem disponível.
# Assim a inferência não falha por uma referência futura ausente na base.
periodo_ref = base.index.max()
periodo_referencia = str(periodo_ref)
print(f"Período de referência selecionado automaticamente: {periodo_referencia}")

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
# 3. Normal expansiva de chuva
# ---------------------------------------------------------------------------
print("\n[4] corrigindo a normal de chuva (normal expansiva, so com o passado)")
ca = base["chuva_acum"].astype(float).values
mes_cal = np.array([p.month for p in base.index])
n_base = len(base)


def normal_expansiva(serie):
    """Media dos mesmos meses do calendario ocorridos antes de cada mes."""
    out = np.full(n_base, np.nan)
    for i in range(n_base):
        ant = [serie[j] for j in range(i) if mes_cal[j] == mes_cal[i] and not np.isnan(serie[j])]
        if ant:
            out[i] = float(np.mean(ant))
    return out


norm_exp = normal_expansiva(ca)
base["chuva_pct_normal_ok"] = np.where(norm_exp > 0, ca / norm_exp * 100, np.nan)
print("    primeiro mes com normal corrigida disponivel:",
      str(base.index[int(np.argmax(~np.isnan(norm_exp)))]))

# ---------------------------------------------------------------------------
# 4. Regua de marcos regulatorios
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


# chuva_pct_normal_ok é a versão corrigida usada pela receita original.
# Na inferência, o parâmetro chuva_pct_normal alimenta essa coluna. A chuva
# acumulada também entra explicitamente no modelo para que o parâmetro
# chuva_acum participe da previsão.
FEATURE_COLS = ["mes_clima", "chuva_acum", "chuva_media", "chuva_pct_normal_ok", "temperatura", "umidade",
                "bandeira_origem", "ear_pct"] + list(MARCOS_4)


def montar(h):
    """Uma linha por origem o; alvo = o+h; usa apenas informacao de o."""
    linhas = []
    for o, r in base.iterrows():
        alvo_mes = o + h
        if alvo_mes not in serie_bandeira.index:
            continue
        d = {"origem": o, "alvo_mes": alvo_mes, "y": int(serie_bandeira.loc[alvo_mes])}
        for c in ["mes_clima", "chuva_acum", "chuva_media", "chuva_pct_normal_ok", "temperatura",
                  "umidade", "bandeira_origem", "ear_pct"]:
            d[c] = r[c]
        linhas.append(d)
    df = pd.DataFrame(linhas)
    alvos = pd.PeriodIndex(df["alvo_mes"])
    for nome in MARCOS_4:
        df[nome] = marca_regime(alvos, nome)
    return df


# ---------------------------------------------------------------------------
# 5. Treino e backtest
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


def treinar(df, cutoff):
    """Treina somente com alvos <= cutoff; não usa o futuro."""
    tr = df[df["alvo_mes"] <= cutoff].copy()
    if len(tr) < MIN_TREINO or tr["y"].nunique() < 2:
        raise ValueError(f"Dados insuficientes para treino até {cutoff}: {len(tr)} linhas")
    cols = [c for c in FEATURE_COLS if c in tr.columns and tr[c].nunique(dropna=True) > 1]
    if not cols:
        raise ValueError(f"Nenhuma variável disponível para treino até {cutoff}")
    m = modelo_final()
    w = peso_amostra(tr["y"].values, tr["alvo_mes"].values)
    m.fit(tr[cols], tr["y"], clf__sample_weight=w)
    return m, cols, len(tr)


def roda(h):
    df = montar(h)
    alvos_teste = [a for a in sorted(df["alvo_mes"].unique()) if a >= INICIO_TESTE]
    yr, yp, meses = [], [], []
    for alvo in alvos_teste:
        lt = df[df["alvo_mes"] == alvo]
        o = lt["origem"].iloc[0]
        try:
            m, usadas, _ = treinar(df, o)
            p = int(m.predict(lt[usadas])[0])
        except Exception:
            continue
        yr.append(int(lt["y"].iloc[0])); yp.append(p); meses.append(alvo)
    return np.array(yr), np.array(yp), meses


print("\n" + "#" * 108)
print("# BACKTEST LONGO: NOTEBOOK 07 (clima + EAR, regua de 4 marcos, peso 5x pos abaco)")
print("#" * 108)
resumo = []
for h in (1,):
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
print("QUADRO RESUMO - NOTEBOOK 07 (clima + EAR, regua de 4 marcos, modelo final)")
print("=" * 100)
print(pd.DataFrame(resumo).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Previsao para o mes solicitado
# ---------------------------------------------------------------------------
# O mês informado é a origem/referência. A previsão t+h mira periodo_ref+h.
# A origem pode ser futura: mes_clima vem do próprio mês e ear_pct é obtido
# para esse mês quando houver observação, sendo imputado pelo pipeline caso
# ainda não exista no histórico.


def construir_linha_futura(origem, h, parametros):
    """Cria a linha de inferência usando os parâmetros informados na origem.

    O EAR e o mês climático são variáveis auxiliares do modelo, pois não fazem
    parte do conjunto de seis parâmetros solicitados. O valor de
    chuva_pct_normal recebido alimenta chuva_pct_normal_ok, que é a variável
    corrigida efetivamente utilizada pela receita do modelo.
    """
    alvo_mes = origem + h
    linha = {"origem": origem, "alvo_mes": alvo_mes}
    linha["mes_clima"] = origem.month
    linha["chuva_acum"] = parametros["chuva_acum"]
    linha["chuva_media"] = parametros["chuva_media"]
    linha["chuva_pct_normal_ok"] = parametros["chuva_pct_normal"]
    linha["temperatura"] = parametros["temperatura"]
    linha["umidade"] = parametros["umidade"]
    linha["bandeira_origem"] = parametros["bandeira_origem"]
    linha["ear_pct"] = float(s_ear.get(origem, np.nan))
    for nome in MARCOS_4:
        linha[nome] = int(marca_regime(pd.PeriodIndex([alvo_mes]), nome)[0])
    return pd.DataFrame([linha])


def prever_horizonte(h, origem, parametros):
    df = montar(h)
    modelo, usadas, n_treino = treinar(df, origem)
    linha = construir_linha_futura(origem, h, parametros)
    classe = int(modelo.predict(linha[usadas])[0])
    probabilidades = modelo.predict_proba(linha[usadas])[0]
    classes = modelo.named_steps["clf"].classes_
    indice_vermelha = int(np.where(classes == 1)[0][0])
    probabilidade_vermelha = float(probabilidades[indice_vermelha])
    probabilidade_nao_vermelha = float(1.0 - probabilidade_vermelha)
    alvo = origem + h
    valor_real = serie_bandeira.get(alvo, np.nan)
    tem_real = not pd.isna(valor_real)
    return {
        "periodo_alvo": str(alvo),
        "classe": classe,
        "classe_nome": "vermelha" if classe == 1 else "nao_vermelha",
        "probabilidade_vermelha": round(probabilidade_vermelha, 6),
        "probabilidade_nao_vermelha": round(probabilidade_nao_vermelha, 6),
        "valor_real": int(valor_real) if tem_real else None,
        "acertou": bool(classe == int(valor_real)) if tem_real else None,
        "n_treino": int(n_treino),
    }


# A interface entrega somente a previsão do horizonte T1.
previsao_t1 = prever_horizonte(1, periodo_ref, parametros_origem)

# Variável com nome explícito para uso no JSON/API.
probabilidade_t1 = previsao_t1["probabilidade_vermelha"]

print("\n### PREVISAO PARAMETRIZADA - T1 ###")
print("referencia simbólica:", periodo_ref)
print("features de origem:", parametros_origem)
print("t+1:", previsao_t1)

# ---------------------------------------------------------------------------
# 7. Peso das variáveis do modelo treinado com toda a base disponível
# ---------------------------------------------------------------------------
print("\n### PESO DAS VARIAVEIS (modelo treinado com toda a base, horizonte t+1) ###")
df1 = montar(1)
cols1 = [c for c in FEATURE_COLS if c in df1.columns and df1[c].nunique(dropna=True) > 1]
m1, cols1, _ = treinar(df1, df1["alvo_mes"].max())
coefs = pd.Series(m1.named_steps["clf"].coef_[0], index=cols1).sort_values(key=abs, ascending=False)
print(coefs.round(3).to_string())

coeficientes = {str(k): float(v) for k, v in coefs.round(6).items()}


# COMMAND ----------

# Restaurar a saída normal e devolver um JSON para a Jobs API.
sys.stdout = _old
_rel = _buf.getvalue()

resultado = {
    "modelo": "07_P_regressao_logistica_clima_ear_regua_final",
    "periodo_referencia": periodo_referencia,
    "features_origem": parametros_origem,
    "previsoes": {
        "t_plus_1": previsao_t1,
    },
    "backtest": resumo,
    "coeficientes": coeficientes,
}
resultado_json = json.dumps(resultado, ensure_ascii=False, default=str)

# Mostra os logs no Databricks e retorna somente JSON pelo get-output.
print(_rel[-12000:])
print("Command skipped$0\n### RESULTADO_API_JSON ###")
print(resultado_json)
dbutils.notebook.exit(resultado_json)