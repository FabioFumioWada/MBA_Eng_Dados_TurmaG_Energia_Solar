# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## Modelo v5 - Bandeira em 3 niveis (Verde/Amarela/Vermelha), com correcoes do Conselho de Modelos
# MAGIC
# MAGIC ### Historico rapido
# MAGIC - **v1** (arvore de decisao) e **v2** (regressao logistica): previam apenas "e vermelha? sim/nao"
# MAGIC   usando clima do mes anterior.
# MAGIC - **v3**: testou EAR (reservatorios) e ONI (indice El Nino/La Nina da NOAA) como variaveis
# MAGIC   extras, ainda no problema binario. Ver `03_regressao_logistica_persistencia_clima_elnino`.
# MAGIC - **v4**: reformulou o problema para 3 classes (Verde/Amarela/Vermelha), horizontes t+1/t+2/t+3.
# MAGIC   Foi submetido a duas rodadas de validacao independente por um Conselho de modelos de IA
# MAGIC   (Claude Fable 5.1 e GPT-5.6 Sol), que encontrou problemas serios de metodologia.
# MAGIC - **v5** (este notebook): aplica TODAS as correcoes obrigatorias apontadas pelo Conselho na
# MAGIC   rodada 3 e testa uma variavel nova recomendada por ambos os modelos (CMO do Sudeste).
# MAGIC
# MAGIC ### Correcoes aplicadas nesta versao (achados do Conselho sobre a v4)
# MAGIC 1. F1-macro da validacao cruzada agora fixa as 3 classes (`labels=[0,1,2]`) em toda chamada -
# MAGIC    antes, folds com uma unica classe presente inflavam artificialmente a metrica.
# MAGIC 2. Selecao do vencedor passa a usar F1-macro sobre as previsoes fora-da-amostra
# MAGIC    CONCATENADAS (pooled OOF), nao a media simples de 5 metricas de distribuicoes desiguais.
# MAGIC 3. Embargo temporal (`gap = h-1`) na validacao cruzada e no corte que antecede o ajuste final,
# MAGIC    para nao vazar rotulos que nao existiriam na data real da previsao em t+2 e t+3.
# MAGIC 4. Holdout avaliado UMA UNICA VEZ para o combo vencedor (decidido so pela CV) e para os
# MAGIC    baselines. A varredura de todas as combinacoes no holdout passa a ser um diagnostico
# MAGIC    exploratorio SEPARADO, sem peso em nenhuma decisao ou conclusao.
# MAGIC 5. Mapeamento de categoria da bandeira e um dicionario explicito com `assert` - categoria
# MAGIC    desconhecida quebra a execucao em vez de virar "Vermelha" silenciosamente.
# MAGIC 6. Teste de significancia reforcado: McNemar pareado (acerto/erro), alem do bootstrap em
# MAGIC    blocos do F1-macro. Diagnosticos ordinais novos: MAE do nivel (0/1/2) e kappa quadratico
# MAGIC    ponderado.
# MAGIC 7. Variavel nova, recomendada pelos dois modelos do Conselho: CMO medio mensal do subsistema
# MAGIC    Sudeste (defasado 1 mes, mesma convencao do EAR) - o custo marginal de operacao reflete
# MAGIC    o estado do sistema de forma mais direta que o EAR isolado.
# MAGIC
# MAGIC ### Correcao de arquitetura de dados (achado do Conselho sobre a v4)
# MAGIC A v4 lia arquivos CSV locais em vez das tabelas tratadas do Unity Catalog. Este notebook
# MAGIC corrige isso: os dados vem de `mba.raw.bandeira_acionada`, `mba.refined.f_modelo_bandeira_clima`,
# MAGIC `mba.trusted.f_ear_subsistema`, `mba.trusted.f_cmo_subsistema` e `mba.trusted.f_oni_enso`.
# MAGIC
# MAGIC ### Base cientifica consultada para esta remodelagem
# MAGIC - O sistema de bandeiras foi criado pela ANEEL em 2015; a bandeira de Escassez Hidrica
# MAGIC   (2021) foi a que teve efeito mensuravel mais forte sobre o consumo (-2,89%) em um estudo
# MAGIC   com dados de 26 estados entre 2015-2022
# MAGIC   ([Silva et al., Energy Economics, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0957178724000444)).
# MAGIC - O ENSO (El Nino/La Nina) tem efeito estimado de ate 11% sobre a Energia Natural Afluente
# MAGIC   (ENA) dependendo da regiao, com o Sudeste mostrando diferenca de +7,3% (43.097 MW) entre
# MAGIC   cenarios de El Nino forte e La Nina forte, usando dados de 1951-2015
# MAGIC   ([Costa & Fernandes, PUC-Rio,
# MAGIC   Maxwell](https://www.maxwell.vrac.puc-rio.br/32290/32290.PDF)) - o que justifica testar o
# MAGIC   ONI como variavel preditiva, como feito desde a v3.
# MAGIC - Em setembro de 2026 o governo brasileiro voltou a reduzir a liberacao de agua dos
# MAGIC   reservatorios e a contratar termeletricas extras por causa de previsao de El Nino mais
# MAGIC   forte ([Valor International,
# MAGIC   2026](https://valorinternational.globo.com/economy/news/2026/08/06/el-nino-forecast-leads-government-to-curtail-hydropower-generation.ghtml)) -
# MAGIC   evidencia de que essa relacao segue ativa e relevante hoje.
# MAGIC - A dificuldade de superar um baseline de persistencia/climatologia em series curtas e um
# MAGIC   resultado classico da meteorologia: a combinacao otima de climatologia e persistencia bate
# MAGIC   qualquer uma das duas isoladas, e ja era descrita na literatura ha mais de 30 anos antes de
# MAGIC   1992 ([Murphy, Weather and Forecasting,
# MAGIC   1992](https://journals.ametsoc.org/view/journals/wefo/7/4/1520-0434_1992_007_0692_cpatlc_2_0_co_2.xml)).
# MAGIC   Isso explica, de forma honesta, por que o resultado deste notebook (persistencia
# MAGIC   competitiva ou melhor que o modelo em alguns horizontes) e esperado e normal, nao um
# MAGIC   fracasso do projeto.
# MAGIC
# MAGIC ### Reprodutibilidade
# MAGIC `random_state` fixo (42). Validacao cruzada temporal (5 janelas, com embargo) para escolher o
# MAGIC vencedor; holdout dos ultimos 24 meses (nunca visto no treino/CV) avaliado uma unica vez.

# COMMAND ----------

# DBTITLE 1,Instala a biblioteca de teste estatistico (McNemar) que nao vem por padrao
# MAGIC %pip install -q statsmodels

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Carrega as tabelas tratadas do Unity Catalog e converte para pandas
from pyspark.sql import functions as F

# Bandeira em 3 niveis a partir da tabela bruta (raw), nao apenas o binario IsVermelha do trusted
bandeira_raw_sp = spark.table("mba.raw.bandeira_acionada").select("DatCompetencia", "NomBandeiraAcionada")
bandeira_raw_sp = bandeira_raw_sp.withColumn("MesCompetencia", F.date_format("DatCompetencia", "yyyyMM").cast("int"))
bandeira_pd = bandeira_raw_sp.select("MesCompetencia", "NomBandeiraAcionada").dropDuplicates(["MesCompetencia"]).toPandas()

clima_pd = spark.table("mba.refined.f_modelo_bandeira_clima").select(
    "MesCompetencia", "Mes", "PrecipitacaoMediaMm", "TemperaturaMediaC", "UmidadeMediaPct"
).toPandas()

print("Colunas EAR:", spark.table("mba.trusted.f_ear_subsistema").columns)

# COMMAND ----------

# DBTITLE 1,Corrige nomes de coluna e agrega EAR/CMO/ONI por mes
from pyspark.sql import functions as F

ear_tbl = spark.table("mba.trusted.f_ear_subsistema")
ear_tbl = ear_tbl.withColumn("MesRef", F.date_format("ear_data", "yyyyMM").cast("int"))
# EAR nacional (SIN) = MWmes verificado / MWmes maximo, somados nos 4 subsistemas e no mes
# (media ponderada pela capacidade de cada subsistema - mesma formula usada no
# EAR consolidado das versoes anteriores, validada batendo com ear_sin_mensal.csv: 60,01%
# em 2024-01 pelas duas contas).
ear_mensal_pd = (
    ear_tbl.groupBy("MesRef")
    .agg(
        F.sum("ear_verif_subsistema_mwmes").alias("_verif"),
        F.sum("ear_max_subsistema").alias("_max"),
    )
    .withColumn("EarPercentualSIN", F.col("_verif") / F.col("_max") * 100)
    .select("MesRef", "EarPercentualSIN")
    .toPandas()
)

cmo_tbl = spark.table("mba.trusted.f_cmo_subsistema").filter(F.col("id_subsistema") == "SE")
cmo_tbl = cmo_tbl.withColumn("MesRef", F.date_format("din_instante", "yyyyMM").cast("int"))
cmo_mensal_pd = (
    cmo_tbl.groupBy("MesRef")
    .agg(F.avg("val_cmomediasemanal").alias("CmoSeMedioMensal"))
    .toPandas()
)

oni_pd = spark.table("mba.trusted.f_oni_enso").select(
    F.col("MesRef").alias("MesRef"), F.col("OniAnomC")
).toPandas()

print("bandeira:", bandeira_pd.shape, "clima:", clima_pd.shape, "ear:", ear_mensal_pd.shape,
      "cmo:", cmo_mensal_pd.shape, "oni:", oni_pd.shape)
display(bandeira_pd.tail())

# COMMAND ----------

# DBTITLE 1,Metodologia v5 completa (mapeamento, features, CV com embargo, holdout unico, testes de significancia)
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, mean_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.contingency_tables import mcnemar

RANDOM_STATE = 42
N_HOLDOUT = 24
rng = np.random.default_rng(RANDOM_STATE)
CLASSES = ["Verde", "Amarela", "Vermelha"]
LABELS = [0, 1, 2]


def add_months(yyyymm, n):
    y, m = divmod(int(yyyymm), 100)
    total = y * 12 + (m - 1) + n
    return (total // 12) * 100 + (total % 12 + 1)


# ---------------------------------------------------------------------------
# 1. Bandeira com 3 niveis - mapeamento EXPLICITO (correcao 5 do Conselho)
# ---------------------------------------------------------------------------
bandeira = bandeira_pd.sort_values("MesCompetencia").reset_index(drop=True)

MAPA_NIVEL = {
    "Verde": 0,
    "Amarela": 1,
    "Vermelha P1": 2,
    "Vermelha P2": 2,
    "Escassez Hídrica": 2,
    "Escassez Hidrica": 2,
}
categorias_fora_do_mapa = set(bandeira["NomBandeiraAcionada"].unique()) - set(MAPA_NIVEL.keys())
assert not categorias_fora_do_mapa, f"Categoria de bandeira desconhecida, corrija o MAPA_NIVEL: {categorias_fora_do_mapa}"
bandeira["NivelBandeira"] = bandeira["NomBandeiraAcionada"].map(MAPA_NIVEL)

print("Distribuicao das 3 classes (todo o periodo, direto do Unity Catalog):")
print(bandeira["NivelBandeira"].map({0: "Verde", 1: "Amarela", 2: "Vermelha"}).value_counts())

clima = clima_pd.sort_values("MesCompetencia").reset_index(drop=True)
clima["MesRef"] = clima["MesCompetencia"].apply(lambda x: add_months(x, -1))
clima["PrecipitacaoMedia3M"] = clima["PrecipitacaoMediaMm"].rolling(3).mean()
clima["TemperaturaMedia3M"] = clima["TemperaturaMediaC"].rolling(3).mean()
clima["MesCompetencia_t1"] = clima["MesCompetencia"]

ear = ear_mensal_pd.copy()
oni = oni_pd.rename(columns={"MesRef": "MesRef"}).copy()
cmo_mensal = cmo_mensal_pd.copy()


def montar_base(h):
    pers = bandeira[["MesCompetencia", "NivelBandeira"]].rename(columns={"MesCompetencia": "MesBase", "NivelBandeira": "NivelBandeiraMesBase"})
    pers["MesCompetencia"] = pers["MesBase"].apply(lambda x: add_months(x, h))

    clima_base = clima.rename(columns={"MesCompetencia_t1": "MesBase"}).copy()
    clima_base["MesCompetencia"] = clima_base["MesBase"].apply(lambda x: add_months(x, h))

    ear_base = ear.copy()
    ear_base["MesBase"] = ear_base["MesRef"].apply(lambda x: add_months(x, 1))
    ear_base["MesCompetencia"] = ear_base["MesBase"].apply(lambda x: add_months(x, h))
    ear_base = ear_base.rename(columns={"EarPercentualSIN": "EarPercentualMesBase"})[["MesCompetencia", "EarPercentualMesBase"]]

    oni_base = oni.copy()
    oni_base["MesBase"] = oni_base["MesRef"].apply(lambda x: add_months(x, 2))
    oni_base["MesCompetencia"] = oni_base["MesBase"].apply(lambda x: add_months(x, h))
    oni_base = oni_base.rename(columns={"OniAnomC": "OniAnomMesBase"})[["MesCompetencia", "OniAnomMesBase"]]

    cmo_base = cmo_mensal.copy()
    cmo_base["MesBase"] = cmo_base["MesRef"].apply(lambda x: add_months(x, 1))
    cmo_base["MesCompetencia"] = cmo_base["MesBase"].apply(lambda x: add_months(x, h))
    cmo_base = cmo_base.rename(columns={"CmoSeMedioMensal": "CmoSeMesBase"})[["MesCompetencia", "CmoSeMesBase"]]

    df = bandeira[["MesCompetencia", "NivelBandeira"]].merge(
        clima_base[["MesCompetencia", "PrecipitacaoMediaMm", "PrecipitacaoMedia3M",
                     "TemperaturaMediaC", "UmidadeMediaPct", "Mes", "TemperaturaMedia3M"]],
        on="MesCompetencia", how="inner"
    )
    df = df.merge(pers[["MesCompetencia", "NivelBandeiraMesBase"]], on="MesCompetencia", how="left")
    df = df.merge(ear_base, on="MesCompetencia", how="left")
    df = df.merge(oni_base, on="MesCompetencia", how="left")
    df = df.merge(cmo_base, on="MesCompetencia", how="left")
    df = df.dropna().sort_values("MesCompetencia").reset_index(drop=True)
    return df


def normal_treino(df, idx_treino):
    base = df.loc[idx_treino]
    n1 = base.groupby("Mes")["PrecipitacaoMediaMm"].mean().rename("PrecipitacaoNormalMm")
    n3 = base.groupby("Mes")["PrecipitacaoMedia3M"].mean().rename("PrecipitacaoNormal3M")
    out = df.merge(n1, on="Mes", how="left").merge(n3, on="Mes", how="left")
    out["PrecipitacaoPctNormal"] = round(out["PrecipitacaoMediaMm"] / out["PrecipitacaoNormalMm"] * 100, 1)
    out["PrecipitacaoPctNormal3M"] = round(out["PrecipitacaoMedia3M"] / out["PrecipitacaoNormal3M"] * 100, 1)
    return out


FEATURE_SETS = {
    "A_base_v1": ["Mes", "PrecipitacaoMediaMm", "PrecipitacaoPctNormal", "TemperaturaMediaC", "UmidadeMediaPct"],
    "B_janela_3_meses": ["Mes", "PrecipitacaoMedia3M", "PrecipitacaoPctNormal3M", "TemperaturaMedia3M", "UmidadeMediaPct"],
    "C_persistencia_nivel": ["Mes", "PrecipitacaoMediaMm", "PrecipitacaoPctNormal", "TemperaturaMediaC", "UmidadeMediaPct", "NivelBandeiraMesBase"],
    "D_tudo_junto": ["Mes", "PrecipitacaoMediaMm", "PrecipitacaoPctNormal", "PrecipitacaoMedia3M",
                      "PrecipitacaoPctNormal3M", "TemperaturaMediaC", "UmidadeMediaPct", "NivelBandeiraMesBase"],
    "E_persistencia_mais_ear": ["Mes", "PrecipitacaoMediaMm", "PrecipitacaoPctNormal", "TemperaturaMediaC",
                                 "UmidadeMediaPct", "NivelBandeiraMesBase", "EarPercentualMesBase"],
    "F_so_ear_mais_persistencia": ["Mes", "NivelBandeiraMesBase", "EarPercentualMesBase"],
    "G_persistencia_mais_oni": ["Mes", "PrecipitacaoMediaMm", "PrecipitacaoPctNormal", "TemperaturaMediaC",
                                 "UmidadeMediaPct", "NivelBandeiraMesBase", "OniAnomMesBase"],
    "H_so_oni_mais_persistencia": ["Mes", "NivelBandeiraMesBase", "OniAnomMesBase"],
    "I_so_persistencia_nivel": ["NivelBandeiraMesBase"],
    "J_persistencia_mais_cmo_se": ["Mes", "PrecipitacaoMediaMm", "PrecipitacaoPctNormal", "TemperaturaMediaC",
                                    "UmidadeMediaPct", "NivelBandeiraMesBase", "CmoSeMesBase"],
    "K_so_cmo_se_mais_persistencia": ["NivelBandeiraMesBase", "CmoSeMesBase"],
    "L_persistencia_ear_cmo_se": ["NivelBandeiraMesBase", "EarPercentualMesBase", "CmoSeMesBase"],
}


def modelos():
    return {
        "ArvoreDecisao": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
        "RegressaoLogistica": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]),
    }


def avaliar(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return dict(
        acuracia=accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        f1_verde=f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0),
        f1_amarela=f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0),
        f1_vermelha=f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0),
        mae_ordinal=mean_absolute_error(y_true, y_pred),
        kappa_quadratico=cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic"),
    )


def bootstrap_bloco_delta_f1macro(y_true, pred_a, pred_b, n_boot=2000, bloco=4):
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    n = len(y_true)
    blocos_idx = [np.arange(i, min(i + bloco, n)) for i in range(0, n, bloco)]
    deltas = []
    for _ in range(n_boot):
        escolhidos = [blocos_idx[i] for i in rng.integers(0, len(blocos_idx), size=len(blocos_idx))]
        idx = np.concatenate(escolhidos)
        f1_a = f1_score(y_true[idx], pred_a[idx], labels=LABELS, average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b[idx], labels=LABELS, average="macro", zero_division=0)
        deltas.append(f1_a - f1_b)
    return np.percentile(deltas, [2.5, 97.5]).tolist(), float(np.mean(deltas))


def mcnemar_acerto_erro(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    acerto_a = (pred_a == y_true)
    acerto_b = (pred_b == y_true)
    n_ab = int(np.sum(acerto_a & ~acerto_b))
    n_ba = int(np.sum(~acerto_a & acerto_b))
    tabela = [[int(np.sum(acerto_a & acerto_b)), n_ab], [n_ba, int(np.sum(~acerto_a & ~acerto_b))]]
    res = mcnemar(tabela, exact=True)
    return dict(estatistica=float(res.statistic), p_valor=float(res.pvalue), n_a_acerta_b_erra=n_ab, n_a_erra_b_acerta=n_ba)


resultados_cv = []
resultados_cv_por_fold = []
resultados_holdout_final = []
resultados_holdout_exploratorio_pos_hoc = []
vencedores = {}

for h in [1, 2, 3]:
    EMBARGO = h - 1
    df_raw = montar_base(h)
    if len(df_raw) <= N_HOLDOUT:
        print(f"t+{h}: dados insuficientes ({len(df_raw)} meses), pulando.")
        continue
    treino_idx_global = df_raw.index[:-N_HOLDOUT]
    treino_idx_fit = treino_idx_global[:-EMBARGO] if EMBARGO > 0 else treino_idx_global

    df_full_cv = normal_treino(df_raw, treino_idx_global)
    df_full_fit = normal_treino(df_raw, treino_idx_fit)

    treino_df = df_full_cv.iloc[:-N_HOLDOUT].reset_index(drop=True)
    treino_df_fit = df_full_fit.iloc[:len(treino_idx_fit)].reset_index(drop=True)
    holdout_df = df_full_fit.iloc[-N_HOLDOUT:].reset_index(drop=True)

    y_treino_full = treino_df["NivelBandeira"]
    y_treino_fit = treino_df_fit["NivelBandeira"]
    y_holdout = holdout_df["NivelBandeira"]

    dist_holdout = y_holdout.map({0: "Verde", 1: "Amarela", 2: "Vermelha"}).value_counts().to_dict()
    print(f"\n=== HORIZONTE t+{h} (embargo={EMBARGO} mes(es)) === treino/CV={len(treino_df)} | "
          f"treino p/ ajuste final={len(treino_df_fit)} | holdout={len(holdout_df)} meses {dist_holdout}")

    tscv = TimeSeriesSplit(n_splits=5, test_size=max(6, len(treino_df) // 10), gap=EMBARGO)

    m_pers_hold = avaliar(y_holdout, holdout_df["NivelBandeiraMesBase"])
    m_maj_hold = avaliar(y_holdout, pd.Series(0, index=holdout_df.index))
    resultados_holdout_final.append(dict(Horizonte=f"t+{h}", Combo="Baseline_Persistencia", **{k: round(v, 3) for k, v in m_pers_hold.items()}))
    resultados_holdout_final.append(dict(Horizonte=f"t+{h}", Combo="Baseline_ClasseMajoritaria", **{k: round(v, 3) for k, v in m_maj_hold.items()}))

    for nome_base in ["Baseline_Persistencia", "Baseline_ClasseMajoritaria"]:
        y_true_oof, y_pred_oof = [], []
        for fold_i, (train_idx, test_idx) in enumerate(tscv.split(treino_df), start=1):
            yt = y_treino_full.iloc[test_idx]
            pt = treino_df["NivelBandeiraMesBase"].iloc[test_idx] if nome_base == "Baseline_Persistencia" else pd.Series(0, index=test_idx)
            y_true_oof.extend(yt.tolist()); y_pred_oof.extend(pt.tolist())
        m_pool = avaliar(y_true_oof, y_pred_oof)
        resultados_cv.append(dict(Horizonte=f"t+{h}", Variaveis=nome_base, Modelo="-",
                                   F1macro_pooled_oof=round(m_pool["f1_macro"], 3), N_meses_oof=len(y_true_oof)))

    melhor_f1_pooled = -1.0
    melhor_combo = None
    for nome_features, features in FEATURE_SETS.items():
        for nome_modelo, modelo in modelos().items():
            y_true_oof, y_pred_oof = [], []
            for fold_i, (train_idx, test_idx) in enumerate(tscv.split(treino_df), start=1):
                df_fold = normal_treino(df_raw.loc[treino_idx_global].reset_index(drop=True), train_idx)
                X_tr = df_fold.iloc[train_idx][features]
                X_te = df_fold.iloc[test_idx][features]
                y_tr = df_fold.iloc[train_idx]["NivelBandeira"]
                y_te = df_fold.iloc[test_idx]["NivelBandeira"]
                if y_tr.nunique() < 2:
                    continue
                modelo.fit(X_tr, y_tr)
                y_pred = modelo.predict(X_te)
                y_true_oof.extend(y_te.tolist()); y_pred_oof.extend(y_pred.tolist())
                resultados_cv_por_fold.append(dict(Horizonte=f"t+{h}", Variaveis=nome_features, Modelo=nome_modelo, Fold=fold_i,
                                                    N_teste=len(test_idx), Verde=int((y_te == 0).sum()), Amarela=int((y_te == 1).sum()), Vermelha=int((y_te == 2).sum())))
            if not y_true_oof:
                continue
            f1_pooled = f1_score(y_true_oof, y_pred_oof, labels=LABELS, average="macro", zero_division=0)
            resultados_cv.append(dict(Horizonte=f"t+{h}", Variaveis=nome_features, Modelo=nome_modelo,
                                       F1macro_pooled_oof=round(f1_pooled, 4), N_meses_oof=len(y_true_oof)))
            if f1_pooled > melhor_f1_pooled:
                melhor_f1_pooled = f1_pooled
                melhor_combo = (nome_features, nome_modelo)

    nome_features, nome_modelo = melhor_combo
    modelo_final = modelos()[nome_modelo]
    modelo_final.fit(treino_df_fit[FEATURE_SETS[nome_features]], y_treino_fit)
    y_pred_hold = modelo_final.predict(holdout_df[FEATURE_SETS[nome_features]])
    m_hold = avaliar(y_holdout, y_pred_hold)
    resultados_holdout_final.append(dict(Horizonte=f"t+{h}", Combo=f"VENCEDOR_CV:{nome_features}/{nome_modelo}", **{k: round(v, 3) for k, v in m_hold.items()}))

    ic95, delta_medio = bootstrap_bloco_delta_f1macro(y_holdout.values, y_pred_hold, holdout_df["NivelBandeiraMesBase"].values)
    delta_observado = m_hold["f1_macro"] - m_pers_hold["f1_macro"]
    mcnemar_res = mcnemar_acerto_erro(y_holdout.values, y_pred_hold, holdout_df["NivelBandeiraMesBase"].values)
    cm_vencedor = confusion_matrix(y_holdout, y_pred_hold, labels=[0, 1, 2])
    cm_persistencia = confusion_matrix(y_holdout, holdout_df["NivelBandeiraMesBase"], labels=[0, 1, 2])

    vencedores[f"t+{h}"] = dict(
        embargo_meses=EMBARGO,
        combo=f"{nome_features}/{nome_modelo}", cv_f1_macro_pooled_oof=round(melhor_f1_pooled, 4),
        holdout=m_hold, baseline_persistencia_holdout=m_pers_hold, baseline_majoritaria_holdout=m_maj_hold,
        delta_f1macro_observado=round(delta_observado, 4),
        delta_f1macro_medio_bootstrap=round(delta_medio, 4), ic95_delta_f1macro=[round(x, 4) for x in ic95],
        mcnemar_vencedor_vs_persistencia=mcnemar_res,
        matriz_confusao_vencedor=cm_vencedor.tolist(), matriz_confusao_persistencia=cm_persistencia.tolist(),
        relatorio_classificacao_vencedor=classification_report(y_holdout, y_pred_hold, labels=[0, 1, 2], target_names=CLASSES, zero_division=0, output_dict=True),
    )
    print(f"Vencedor (CV, F1macro pooled OOF={melhor_f1_pooled:.4f}): {nome_features}/{nome_modelo}")
    print(f"  Holdout vencedor: {m_hold}")
    print(f"  Holdout persistencia: {m_pers_hold}")
    print(f"  Delta F1macro observado={delta_observado:.4f} | bootstrap medio={delta_medio:.4f} | IC95%={ic95}")
    print(f"  McNemar vencedor vs persistencia: {mcnemar_res}")

    for nome_features2, features2 in FEATURE_SETS.items():
        for nome_modelo2, modelo2 in modelos().items():
            modelo2.fit(treino_df_fit[features2], y_treino_fit)
            yp = modelo2.predict(holdout_df[features2])
            mh = avaliar(y_holdout, yp)
            resultados_holdout_exploratorio_pos_hoc.append(dict(Horizonte=f"t+{h}", Variaveis=nome_features2, Modelo=nome_modelo2, **{k: round(v, 3) for k, v in mh.items()}))

print("\n\n=== RESUMO FINAL MULTICLASSE v5 (dados direto do Unity Catalog) ===")
for h, v in vencedores.items():
    print(f"\n{h} (embargo={v['embargo_meses']} mes(es)): {v['combo']} (F1macro CV pooled OOF={v['cv_f1_macro_pooled_oof']})")
    print(f"  Holdout vencedor: {v['holdout']}")
    print(f"  Persistencia holdout: {v['baseline_persistencia_holdout']}")
    print(f"  Delta F1macro observado={v['delta_f1macro_observado']} | bootstrap medio={v['delta_f1macro_medio_bootstrap']} IC95%={v['ic95_delta_f1macro']}")
    print(f"  McNemar: {v['mcnemar_vencedor_vs_persistencia']}")

# COMMAND ----------

# DBTITLE 1,Salva os resultados como tabela nova no Unity Catalog (refined) para consulta futura
cv_sp = spark.createDataFrame(pd.DataFrame(resultados_cv))
holdout_sp = spark.createDataFrame(pd.DataFrame(resultados_holdout_final))

cv_sp.write.mode("overwrite").saveAsTable("mba.refined.resultado_cv_modelo_v5")
holdout_sp.write.mode("overwrite").saveAsTable("mba.refined.resultado_holdout_modelo_v5")

print("Tabelas novas salvas: mba.refined.resultado_cv_modelo_v5 e mba.refined.resultado_holdout_modelo_v5")
display(holdout_sp)
