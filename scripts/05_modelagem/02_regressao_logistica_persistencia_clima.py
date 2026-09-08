# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Modelo v2 corrigido: regressão logística, persistência e clima
# MAGIC
# MAGIC Esta revisão substitui a avaliação binária antiga, não o modelo v5.
# MAGIC **Proposta preservada:** regressão logística com clima e bandeira conhecida na origem.
# MAGIC
# MAGIC ## Regras iguais nas três versões
# MAGIC - Classes: Verde=0, Amarela=1, Vermelha=2. Os dois patamares vermelhos
# MAGIC   e Escassez Hídrica são agrupados em 2, uma simplificação documentada.
# MAGIC - Em cada origem t, a bandeira de t já é conhecida. Prever t+1, t+2 e t+3.
# MAGIC - Clima de t-1; ONI com mês central t-2. Não usar clima observado do futuro.
# MAGIC - Os últimos 24 meses-alvo ficam fora do treino/CV. O mesmo calendário
# MAGIC   de teste é exigido nos três horizontes e nas três versões.
# MAGIC - Cinco janelas temporais expansivas, gap=h-1, e verificação explícita
# MAGIC   de que o último rótulo de treino existe na primeira origem de teste.
# MAGIC - Normais de chuva e escalonamento aprendidos só no treino de cada janela.
# MAGIC   A normal é uma média sazonal amostral, não uma normal climatológica oficial.
# MAGIC - F1-macro pooled OOF com labels=[0,1,2], persistência e classe majoritária
# MAGIC   calculada SOMENTE no treino. Não há busca de novo algoritmo nesta revisão.
# MAGIC - Cronômetro igual: perf_counter ao redor de fit. Inclui StandardScaler.fit
# MAGIC   na regressão; exclui leitura, preparo da base, previsão e estatística.
# MAGIC - Uma avaliação por horizonte nesta execução. Como o período final já foi
# MAGIC   examinado em versões anteriores, ele é um backtest revisitado, não uma
# MAGIC   nova confirmação independente. Não escolher vencedor pelo holdout.
# MAGIC
# MAGIC ## Limites que permanecem
# MAGIC A chuva média reduz o efeito da soma de estações, mas a composição da rede
# MAGIC ainda pode mudar. ONI/clima usam séries históricas atuais, sem vintages:
# MAGIC o atraso é uma hipótese, não prova de disponibilidade em cada data passada.
# MAGIC McNemar compara acertos, NÃO a diferença de F1. O bootstrap em blocos dá
# MAGIC um intervalo exploratório dessa diferença; 24 meses é uma amostra pequena.
# MAGIC Estes são testes históricos com origens móveis, não previsões futuras oficiais.
# MAGIC O modelo final fica congelado durante o teste; as entradas são atualizadas
# MAGIC a cada origem. Não é uma previsão simultânea de 24 meses a partir de uma data.
# MAGIC As tabelas de origem e o notebook 04/v5 são apenas preservados, nunca escritos.

# COMMAND ----------

# DBTITLE 1,Configuração e bibliotecas
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, mean_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

VERSAO = "v2"
SEED = 42
N_HOLDOUT = 24
N_FOLDS = 5
LABELS = [0, 1, 2]
CLASSES = ["Verde", "Amarela", "Vermelha"]
MAPA_NIVEL = {
    "Verde": 0, "Amarela": 1, "Vermelha P1": 2, "Vermelha P2": 2,
    "Escassez Hídrica": 2, "Escassez Hidrica": 2,
}
FEATURES_CLIMA = [
    "MesClima", "PrecipitacaoMediaMm", "PrecipitacaoPctNormalTreino",
    "TemperaturaMediaC", "UmidadeMediaPct",
]


def adicionar_meses(mes, n):
    ano, mes_ano = divmod(int(mes), 100)
    assert 1 <= mes_ano <= 12, f"Mês inválido: {mes}"
    total = ano * 12 + mes_ano - 1 + n
    return (total // 12) * 100 + total % 12 + 1


def exigir_meses_unicos(df, coluna, nome):
    assert df[coluna].notna().all(), f"{nome}: mês nulo."
    assert not df[coluna].duplicated().any(), f"{nome}: mês duplicado."
    df[coluna] = pd.to_numeric(df[coluna], errors="raise").astype(int)
    for mes in df[coluna]:
        adicionar_meses(mes, 0)


def exigir_continuidade(meses):
    valores = list(map(int, meses))
    assert all(b == adicionar_meses(a, 1) for a, b in zip(valores, valores[1:])), (
        "Há lacuna mensal: corrigir a fonte, não tratar linhas como meses consecutivos.")


def preparar_fontes(bandeira_entrada, clima_entrada, oni_entrada):
    b = bandeira_entrada.copy()
    c = clima_entrada.copy()
    o = oni_entrada.copy()
    exigir_meses_unicos(b, "MesCompetencia", "Bandeira")
    exigir_meses_unicos(c, "MesCompetencia", "Clima refined")
    exigir_meses_unicos(o, "MesRef", "ONI")
    assert b["NomBandeiraAcionada"].notna().all(), "Bandeira sem categoria."
    categorias = b["NomBandeiraAcionada"].astype(str).str.strip()
    desconhecidas = set(categorias) - set(MAPA_NIVEL)
    assert not desconhecidas, f"Categoria não mapeada: {desconhecidas}"
    b["NivelBandeira"] = categorias.map(MAPA_NIVEL).astype(int)
    b = b.sort_values("MesCompetencia").reset_index(drop=True)
    exigir_continuidade(b["MesCompetencia"])
    # O refined tem clima de MesCompetencia-1. Não aplicar lag por posição.
    assert c["MesReferenciaClima"].notna().all()
    c["MesReferenciaClima"] = c["MesReferenciaClima"].astype(int)
    assert (c["MesReferenciaClima"] ==
            c["MesCompetencia"].map(lambda x: adicionar_meses(x, -1))).all(), (
        "Convenção de datas do refined mudou: interromper para revisar.")
    c["MesBase"] = c["MesCompetencia"]
    c["MesClima"] = c["MesReferenciaClima"] % 100
    for col in ["PrecipitacaoMediaMm", "TemperaturaMediaC", "UmidadeMediaPct"]:
        c[col] = pd.to_numeric(c[col], errors="raise")
    o["OniAnomC"] = pd.to_numeric(o["OniAnomC"], errors="raise")
    return b, c, o


def montar_base(b, c, o, h):
    df = c[["MesBase", "MesReferenciaClima", "MesClima", "PrecipitacaoMediaMm",
            "TemperaturaMediaC", "UmidadeMediaPct"]].copy()
    df["MesAlvo"] = df["MesBase"].map(lambda x: adicionar_meses(x, h))
    df = df.merge(
        b[["MesCompetencia", "NivelBandeira"]].rename(columns={
            "MesCompetencia": "MesBase", "NivelBandeira": "NivelBandeiraMesBase"}),
        on="MesBase", how="left", validate="one_to_one")
    df = df.merge(
        b[["MesCompetencia", "NivelBandeira"]].rename(columns={"MesCompetencia": "MesAlvo"}),
        on="MesAlvo", how="inner", validate="one_to_one")
    df["MesRefONI"] = df["MesBase"].map(lambda x: adicionar_meses(x, -2))
    df = df.merge(o[["MesRef", "OniAnomC"]].rename(columns={"MesRef": "MesRefONI"}),
                  on="MesRefONI", how="left", validate="many_to_one")
    # Interseção comum às três versões: ONI é feature SOMENTE da v3.
    # Isso impede v1/v2/v3 serem avaliadas em amostras diferentes silenciosamente.
    obrigatorias = ["PrecipitacaoMediaMm", "TemperaturaMediaC", "UmidadeMediaPct",
                    "NivelBandeiraMesBase", "NivelBandeira", "OniAnomC"]
    faltantes = df.loc[df[obrigatorias].isna().any(axis=1), "MesAlvo"].astype(int).tolist()
    df = df.dropna(subset=obrigatorias).sort_values("MesAlvo").reset_index(drop=True)
    assert np.isfinite(df[obrigatorias].to_numpy(dtype=float)).all()
    df["NivelBandeiraMesBase"] = df["NivelBandeiraMesBase"].astype(int)
    df["NivelBandeira"] = df["NivelBandeira"].astype(int)
    exigir_continuidade(df["MesAlvo"])
    assert (df["MesReferenciaClima"] < df["MesBase"]).all()
    assert (df["MesRefONI"] < df["MesReferenciaClima"]).all()
    assert (df["MesBase"] < df["MesAlvo"]).all()
    return df, faltantes


def features(versao):
    cols = list(FEATURES_CLIMA)
    if versao in ("v2", "v3"):
        cols += ["NivelBandeiraMesBase"]
    if versao == "v3":
        cols += ["OniAnomC"]
    assert versao in ("v1", "v2", "v3")
    return cols


def calcular_features_treino(treino, avaliacao, versao):
    """Somente o treino define a referência; não modifica as tabelas de origem."""
    normais = treino.groupby("MesClima")["PrecipitacaoMediaMm"].mean()
    assert len(normais) == 12 and (normais > 0).all(), (
        "Treino sem 12 meses sazonais válidos ou referência de chuva zero.")

    def transformar(df):
        out = df.copy()
        out["PrecipitacaoPctNormalTreino"] = (
            100 * out["PrecipitacaoMediaMm"] / out["MesClima"].map(normais)).round(1)
        X = out[features(versao)]
        assert np.isfinite(X.to_numpy(dtype=float)).all(), "Feature inválida."
        return X

    return transformar(treino), transformar(avaliacao), normais.to_dict()


def criar_modelo(versao):
    if versao == "v1":
        return DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
    ])


def ajustar_cronometrado(modelo, X, y):
    assert pd.Series(y).nunique() >= 2, "Treino com uma classe: não pular fold silenciosamente."
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        inicio = time.perf_counter()
        modelo.fit(X, y)
        duracao = (time.perf_counter() - inicio) * 1000
    return duracao


def avaliar(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    f1_classes = f1_score(y, pred, labels=LABELS, average=None, zero_division=0)
    kappa = (float(cohen_kappa_score(y, pred, labels=LABELS, weights="quadratic"))
             if len(np.unique(np.concatenate([y, pred]))) > 1 else None)
    return {
        "Acuracia": float(accuracy_score(y, pred)),
        "F1macro": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "F1verde": float(f1_classes[0]), "F1amarela": float(f1_classes[1]),
        "F1vermelha": float(f1_classes[2]),
        "MAEordinal": float(mean_absolute_error(y, pred)),
        "KappaQuadratico": kappa,
    }


def comparar_persistencia(y, pred, persistencia, h, n_boot=2000):
    """McNemar exato por binomial; bootstrap circular em blocos de quatro meses."""
    y, pred, persistencia = map(np.asarray, (y, pred, persistencia))
    so_modelo = int(np.sum((pred == y) & (persistencia != y)))
    so_persistencia = int(np.sum((pred != y) & (persistencia == y)))
    discordantes = so_modelo + so_persistencia
    p = float(binomtest(so_modelo, discordantes, .5).pvalue) if discordantes else 1.0
    rng = np.random.default_rng(SEED + h)
    n, bloco = len(y), 4
    deltas = np.empty(n_boot)
    # F1 via matriz 3x3: a mesma definição sklearn, acelerada para bootstrap.
    def f1_fixo(a, b):
        cm = np.bincount(a * 3 + b, minlength=9).reshape(3, 3)
        denom = cm.sum(axis=0) + cm.sum(axis=1)
        scores = np.divide(2 * cm.diagonal(), denom, out=np.zeros(3), where=denom != 0)
        return scores.mean()
    for i in range(n_boot):
        inicios = rng.integers(0, n, size=int(np.ceil(n / bloco)))
        idx = ((inicios[:, None] + np.arange(bloco)) % n).ravel()[:n]
        deltas[i] = f1_fixo(y[idx], pred[idx]) - f1_fixo(y[idx], persistencia[idx])
    ic = np.quantile(deltas, [.025, .975])
    return {
        "McNemarP": p, "SoModeloAcerta": so_modelo, "SoPersistenciaAcerta": so_persistencia,
        "DeltaF1macro": float(f1_fixo(y, pred) - f1_fixo(y, persistencia)),
        "IC95DeltaF1inf": float(ic[0]), "IC95DeltaF1sup": float(ic[1]),
        "BootstrapRepeticoes": n_boot, "BootstrapBlocoMeses": bloco,
    }


def verificar_corte(treino, avaliacao):
    assert treino["MesAlvo"].max() < avaliacao["MesAlvo"].min()
    assert treino["MesAlvo"].max() <= avaliacao["MesBase"].min(), (
        "Vazamento: rótulo de treino posterior à primeira origem de avaliação.")


def extrair_coeficientes(modelo, versao, horizonte):
    clf = modelo.named_steps["clf"]
    if len(clf.classes_) == 2:
        linhas = [(int(clf.classes_[1]), clf.coef_[0],
                   "versus " + CLASSES[int(clf.classes_[0])])]
    else:
        linhas = [(int(classe), linha, "escore multiclasse; sem interpretação causal")
                  for classe, linha in zip(clf.classes_, clf.coef_)]
    return pd.DataFrame([
        {"Horizonte": horizonte, "Classe": CLASSES[classe], "Contraste": contraste,
         "Variavel": feature, "CoeficientePadronizado": float(coef)}
        for classe, linha, contraste in linhas
        for feature, coef in zip(features(versao), linha)
    ])


def executar(bandeira_entrada, clima_entrada, oni_entrada, versao):
    inicio_avaliacao = time.perf_counter()
    b, c, o = preparar_fontes(bandeira_entrada, clima_entrada, oni_entrada)
    calendario_holdout = b["MesCompetencia"].tail(N_HOLDOUT).astype(int).tolist()
    exigir_continuidade(calendario_holdout)
    resumo, cv, folds, previsoes, detalhes, artefatos = [], [], [], [], {}, {}
    for h in [1, 2, 3]:
        df, faltantes = montar_base(b, c, o, h)
        assert len(df) >= 84, "Poucos meses: manter 24 de holdout e pelo menos 60 anteriores."
        holdout = df.tail(N_HOLDOUT).copy()
        assert holdout["MesAlvo"].astype(int).tolist() == calendario_holdout, (
            "Faltam features nos 24 meses finais. Não deslocar o teste para esconder ausência.")
        pre = df.iloc[:-N_HOLDOUT].copy().reset_index(drop=True)
        gap = h - 1
        splits = list(TimeSeriesSplit(
            n_splits=N_FOLDS, test_size=max(6, len(pre) // 10), gap=gap).split(pre))
        oof_y = []
        oof = {"Modelo": [], "Persistencia": [], "ClasseMajoritaria": []}
        tempos_cv = []
        for fold, (tr, va) in enumerate(splits, 1):
            treino, validacao = pre.iloc[tr], pre.iloc[va]
            verificar_corte(treino, validacao)
            Xtr, Xva, normais = calcular_features_treino(treino, validacao, versao)
            modelo = criar_modelo(versao)
            tempo = ajustar_cronometrado(modelo, Xtr, treino["NivelBandeira"])
            tempos_cv.append(tempo)
            pred = modelo.predict(Xva).astype(int)
            maioria = int(treino["NivelBandeira"].mode().iloc[0])
            oof_y.extend(validacao["NivelBandeira"].astype(int).tolist())
            oof["Modelo"].extend(pred.tolist())
            oof["Persistencia"].extend(validacao["NivelBandeiraMesBase"].astype(int).tolist())
            oof["ClasseMajoritaria"].extend([maioria] * len(validacao))
            folds.append({
                "Versao": versao, "Horizonte": f"t+{h}", "Fold": fold, "GapMeses": gap,
                "TreinoInicio": int(treino["MesAlvo"].min()),
                "TreinoFim": int(treino["MesAlvo"].max()),
                "ValidacaoInicio": int(validacao["MesAlvo"].min()),
                "ValidacaoFim": int(validacao["MesAlvo"].max()),
                "PrimeiraOrigemValidacao": int(validacao["MesBase"].min()),
                "Ntreino": len(treino), "Nvalidacao": len(validacao), "FitMs": tempo,
            })
        for metodo, preds in oof.items():
            cv.append({"Versao": versao, "Horizonte": f"t+{h}", "Metodo": metodo,
                       "NmesesOOF": len(oof_y), **avaliar(oof_y, preds)})

        treino_final = pre.iloc[:-gap] if gap else pre
        verificar_corte(treino_final, holdout)
        Xtr, Xho, normais = calcular_features_treino(treino_final, holdout, versao)
        modelo_final = criar_modelo(versao)
        fit_ms = ajustar_cronometrado(modelo_final, Xtr, treino_final["NivelBandeira"])
        inicio_pred = time.perf_counter()
        pred = modelo_final.predict(Xho).astype(int)
        pred_ms = (time.perf_counter() - inicio_pred) * 1000
        y = holdout["NivelBandeira"].to_numpy(dtype=int)
        pers = holdout["NivelBandeiraMesBase"].to_numpy(dtype=int)
        maioria = int(treino_final["NivelBandeira"].mode().iloc[0])
        preds = {"Modelo": pred, "Persistencia": pers,
                 "ClasseMajoritaria": np.full(len(y), maioria, dtype=int)}
        assinatura = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
        for metodo, p in preds.items():
            resumo.append({
                "Versao": versao, "Horizonte": f"t+{h}", "Metodo": metodo,
                "TreinoInicio": int(treino_final["MesAlvo"].min()),
                "TreinoFim": int(treino_final["MesAlvo"].max()),
                "TesteInicio": int(holdout["MesAlvo"].min()),
                "TesteFim": int(holdout["MesAlvo"].max()),
                "Ntreino": len(treino_final), "Nteste": len(holdout), "GapMeses": gap,
                **avaliar(y, p),
            })
        comparacao = comparar_persistencia(y, pred, pers, h)
        detalhes[f"t+{h}"] = {
            "Features": features(versao), "FitFinalMs": fit_ms,
            "PredictHoldoutMs": pred_ms, "FitCVTotalMs": float(sum(tempos_cv)),
            "FitCVPorFoldMs": tempos_cv, "NormaisAprendidasTreino": normais,
            "ClasseMajoritariaTreino": CLASSES[maioria], "BaseSHA256": assinatura,
            "MesesExcluidosPorAusencia": faltantes, "MesesTotaisElegiveis": len(df),
            "MatrizModelo": confusion_matrix(y, pred, labels=LABELS).tolist(),
            "MatrizPersistencia": confusion_matrix(y, pers, labels=LABELS).tolist(),
            "RelatorioClasses": classification_report(
                y, pred, labels=LABELS, target_names=CLASSES, zero_division=0, output_dict=True),
            **comparacao,
        }
        for i, row in enumerate(holdout.to_dict("records")):
            previsoes.append({
                "Versao": versao, "Horizonte": f"t+{h}",
                **{k: int(row[k]) for k in ["MesBase", "MesAlvo", "MesReferenciaClima", "MesRefONI"]},
                "BandeiraConhecida": CLASSES[int(row["NivelBandeiraMesBase"])],
                "Real": CLASSES[int(y[i])], "Previsto": CLASSES[int(pred[i])],
                "Persistencia": CLASSES[int(pers[i])],
                "AcertouModelo": bool(y[i] == pred[i]), "AcertouPersistencia": bool(y[i] == pers[i]),
            })
        artefatos[f"t+{h}"] = modelo_final
    payload = {
        "Versao": versao, "ExecutadoUTC": datetime.now(timezone.utc).isoformat(),
        "Ambiente": {"Python": platform.python_version(), "Sklearn": sklearn.__version__,
                     "Pandas": pd.__version__, "Numpy": np.__version__},
        "Protocolo": "3 classes; t+1/t+2/t+3; gap=h-1; CV temporal; backtest revisitado 24 meses",
        "TempoAvaliacaoTotalMs": (time.perf_counter() - inicio_avaliacao) * 1000,
        "ResumoHoldout": resumo, "ResumoCV": cv, "AuditoriaFolds": folds,
        "PrevisoesHoldout": previsoes, "Detalhes": detalhes,
    }
    return payload, artefatos

# COMMAND ----------

# DBTITLE 1,Leitura real do Unity Catalog, sem escrever nas tabelas de origem
if "spark" in globals():
    from pyspark.sql import functions as F
    inicio_leitura = time.perf_counter()
    bandeira_pd = spark.table("mba.raw.bandeira_acionada").select(
        F.date_format("DatCompetencia", "yyyyMM").cast("int").alias("MesCompetencia"),
        "NomBandeiraAcionada",
    ).toPandas()
    clima_pd = spark.table("mba.refined.f_modelo_bandeira_clima").select(
        "MesCompetencia", "MesReferenciaClima", "PrecipitacaoMediaMm",
        "TemperaturaMediaC", "UmidadeMediaPct",
    ).toPandas()
    oni_pd = spark.table("mba.trusted.f_oni_enso").select("MesRef", "OniAnomC").toPandas()
    tempo_leitura_ms = (time.perf_counter() - inicio_leitura) * 1000
    resultado, modelos_ajustados = executar(bandeira_pd, clima_pd, oni_pd, VERSAO)
    resultado["TempoLeituraMs"] = tempo_leitura_ms
    resultado["OrigemExecucao"] = "Databricks pessoal; Unity Catalog mba"
    resumo_holdout = pd.DataFrame(resultado["ResumoHoldout"])
    resumo_cv = pd.DataFrame(resultado["ResumoCV"])
    previsoes_holdout = pd.DataFrame(resultado["PrevisoesHoldout"])
    auditoria_folds = pd.DataFrame(resultado["AuditoriaFolds"])
    tempos = pd.DataFrame([
        {"Versao": VERSAO, "Horizonte": h,
         **{k: d[k] for k in ["FitFinalMs", "FitCVTotalMs", "PredictHoldoutMs"]}}
        for h, d in resultado["Detalhes"].items()
    ])
    display(resumo_cv)
    display(resumo_holdout)
    display(tempos)
    display(previsoes_holdout)
    display(auditoria_folds)
    print("McNemar é sobre acerto/erro, não um teste de F1-macro.")
    for h, d in resultado["Detalhes"].items():
        print(h, "delta F1:", d["DeltaF1macro"],
              "IC95 bootstrap:", [d["IC95DeltaF1inf"], d["IC95DeltaF1sup"]],
              "p McNemar exploratório:", d["McNemarP"])

# COMMAND ----------

# DBTITLE 1,Matrizes de confusão e interpretação do modelo
if "spark" in globals():
    import matplotlib.pyplot as plt
    for h, d in resultado["Detalhes"].items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, campo, titulo in zip(axes, ["MatrizModelo", "MatrizPersistencia"],
                                     [f"{VERSAO} {h}", "Persistência"]):
            cm = np.array(d[campo])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
            ax.set(xticks=LABELS, yticks=LABELS, xticklabels=CLASSES,
                   yticklabels=CLASSES, xlabel="Previsto", ylabel="Real", title=titulo)
            for i in LABELS:
                for j in LABELS:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.suptitle("Backtest de 24 meses; classes fixas")
        fig.tight_layout()
        display(fig)
        plt.close(fig)
        modelo = modelos_ajustados[h]
        if VERSAO == "v1":
            print(h, "\n", export_text(modelo, feature_names=features(VERSAO)))
            fig, ax = plt.subplots(figsize=(15, 7))
            plot_tree(modelo, feature_names=features(VERSAO),
                      class_names=[CLASSES[int(x)] for x in modelo.classes_],
                      filled=True, rounded=True, fontsize=8, ax=ax)
            ax.set_title(f"Árvore v1 corrigida: {h}")
            fig.tight_layout()
            display(fig)
            plt.close(fig)
        else:
            coeficientes = extrair_coeficientes(modelo, VERSAO, h)
            print("Coeficientes do classificador nominal; não demonstram causalidade.")
            display(coeficientes)

# COMMAND ----------

# DBTITLE 1,Saída auditável da execução
if "spark" in globals():
    # Sem salvar tabela, sem retreinar sobre o holdout e sem tocar no v5.
    dbutils.notebook.exit(json.dumps(resultado, ensure_ascii=False, allow_nan=False))
