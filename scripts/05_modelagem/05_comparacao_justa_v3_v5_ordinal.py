# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Comparação justa: v3, v5 e regressão ordinal
# MAGIC
# MAGIC Notebook novo. Não modifica notebooks anteriores, tabelas, Git nem agendamentos.
# MAGIC Verde=0, Amarela=1, Vermelha=2; P1/P2/Escassez Hídrica agrupadas.
# MAGIC Em cada origem t, a bandeira de t é conhecida. Prever t+1, t+2, t+3.
# MAGIC Clima e CMO de t-1; ONI central t-2; médias móveis só olham para trás.
# MAGIC Mesmos meses-alvo de teste e mesma base completa para todos os candidatos.
# MAGIC Treino, seleção e normais sazonais usam somente informação disponível na
# MAGIC primeira origem do teste. CV expansiva com gap=h-1, cinco janelas.
# MAGIC Para h>1 o embargo é aplicado ANTES da CV e da seleção, não só no fit final.
# MAGIC
# MAGIC ## Candidatos fixados antes desta execução
# MAGIC - v3: configuração corrigida com clima, bandeira conhecida e ONI.
# MAGIC - v5: receitas selecionadas anteriormente, reavaliadas sem nova busca:
# MAGIC   t+1 logística balanceada com CMO e bandeira; t+2 árvore com clima de 3 meses;
# MAGIC   t+3 árvore com clima de 1 mês. Não reproduz os resultados antigos literalmente.
# MAGIC - Ordinal: OrderedModel logit, CMO e bandeira conhecida, sem pesos nem penalização.
# MAGIC - Controle nominal: mesmas duas entradas, sem pesos nem penalização, para
# MAGIC   distinguir o efeito de modelar a ordem do efeito das entradas/pesos/penalização.
# MAGIC - Referências: persistência e classe majoritária aprendida no treino.
# MAGIC
# MAGIC A ordinal supõe chances proporcionais e efeitos comuns nos dois limiares.
# MAGIC A bandeira conhecida entra como escore 0/1/2, uma hipótese linear na entrada.
# MAGIC Não há validação formal dessa hipótese nem garantia de melhor desempenho.
# MAGIC F1-macro pooled OOF é o critério principal; MAE ordinal é diagnóstico de
# MAGIC distância em categorias, não erro monetário. Desempate: menor MAE, depois regra simples.
# MAGIC Escolha congelada pela CV ANTES de medir o teste. Não escolher pela tabela de teste.
# MAGIC
# MAGIC ## Limitações e fontes
# MAGIC Os 24 meses finais já foram examinados: este é um backtest revisitado,
# MAGIC não uma confirmação nova. As receitas também refletem pesquisa anterior.
# MAGIC Séries históricas atuais, sem versões de publicação: os lags são hipóteses
# MAGIC de disponibilidade, não prova de disponibilidade pontual no passado.
# MAGIC A média mensal do CMO conserva a regra anterior: média aritmética dos
# MAGIC registros semanais classificados pelo mês de din_instante, não média diária ponderada.
# MAGIC A chuva é média por estação; a composição da rede ainda pode variar.
# MAGIC Fit inclui scaler e ajuste, não leitura/preparo/predict/estatística; tempos são
# MAGIC de uma execução, não benchmark. O modelo fica congelado; entradas mudam por origem.
# MAGIC IC bootstrap circular (blocos de 4 meses, 2000 réplicas) e McNemar são exploratórios.
# MAGIC McNemar compara acertos, não F1. Não há correção por múltiplas comparações.
# MAGIC Referências:
# MAGIC https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html
# MAGIC https://scikit-learn.org/stable/modules/cross_validation.html
# MAGIC
# COMMAND ----------
# DBTITLE 1,Funções de datas, fontes, métricas e segurança reaproveitadas
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

VERSAO = "comparacao_ordinal"
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


# COMMAND ----------
# DBTITLE 1,Regressão ordinal e candidatos fixados
import scipy
import statsmodels
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import log_loss
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning
from statsmodels.tools.sm_exceptions import HessianInversionWarning

CANDIDATOS = ["v3", "v5_receita_fixa", "Ordinal_CMO", "Controle_nominal_CMO"]
REFERENCIAS = ["Persistencia", "ClasseMajoritaria"]
PRIORIDADE = {name: i for i, name in enumerate(REFERENCIAS + [
    "Ordinal_CMO", "Controle_nominal_CMO", "v5_receita_fixa", "v3"])}


class LogisticaOrdinal:
    """Chances proporcionais, sem intercepto explícito, pesos ou regularização."""
    def fit(self, X, y):
        assert set(np.asarray(y, dtype=int)) == set(LABELS), (
            "Ordinal requer as três classes neste experimento; não omitir folds.")
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        assert (np.std(Z, axis=0) > 1e-10).all(), "Entrada constante na ordinal."
        self.model_ = OrderedModel(
            pd.Series(pd.Categorical(np.asarray(y), categories=LABELS, ordered=True)),
            Z, distr="logit")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SMConvergenceWarning)
            warnings.simplefilter("error", HessianInversionWarning)
            self.result_ = self.model_.fit(
                method="bfgs", maxiter=1000, disp=False, gtol=1e-7)
        assert self.result_.mle_retvals["converged"], "Ordinal não convergiu."
        assert np.isfinite(np.asarray(self.result_.params)).all()
        self.classes_ = np.asarray(LABELS)
        return self

    def predict_proba(self, X):
        p = np.asarray(self.model_.predict(
            self.result_.params, exog=self.scaler_.transform(X)), dtype=float)
        assert p.shape == (len(X), 3)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        assert np.allclose(p.sum(axis=1), 1)
        return p

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(int)


def colunas_candidato(nome, h):
    if nome == "v3":
        return features("v3")
    if nome == "v5_receita_fixa" and h in (2, 3):
        return (["MesClima", "PrecipitacaoMedia3M", "PrecipitacaoPctNormal3M",
                 "TemperaturaMedia3M", "UmidadeMediaPct"] if h == 2 else FEATURES_CLIMA)
    return ["NivelBandeiraMesBase", "CmoSeMesBase"]


def estimador_candidato(nome, h):
    if nome == "Ordinal_CMO":
        return LogisticaOrdinal()
    if nome == "Controle_nominal_CMO":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty=None, class_weight=None,
                                       max_iter=1000, random_state=SEED)),
        ])
    if nome == "v5_receita_fixa" and h in (2, 3):
        return criar_modelo("v1")
    return criar_modelo("v3")


def preparar_bases_comuns(bandeiras, clima, oni, cmo):
    b, c, o = preparar_fontes(bandeiras, clima, oni)
    c = c.sort_values("MesBase").reset_index(drop=True)
    exigir_continuidade(c["MesBase"])
    c["PrecipitacaoMedia3M"] = c["PrecipitacaoMediaMm"].rolling(3, min_periods=3).mean()
    c["TemperaturaMedia3M"] = c["TemperaturaMediaC"].rolling(3, min_periods=3).mean()
    m = cmo.copy()
    exigir_meses_unicos(m, "MesRef", "CMO mensal")
    m["CmoSeMedioMensal"] = pd.to_numeric(m["CmoSeMedioMensal"], errors="raise")
    m = m.rename(columns={"MesRef": "MesRefCMO", "CmoSeMedioMensal": "CmoSeMesBase"})
    bases, exclusoes = {}, {}
    for h in [1, 2, 3]:
        df, faltantes = montar_base(b, c, o, h)
        df = df.merge(c[["MesBase", "PrecipitacaoMedia3M", "TemperaturaMedia3M"]],
                      on="MesBase", how="left", validate="one_to_one")
        df["MesRefCMO"] = df["MesBase"].map(lambda x: adicionar_meses(x, -1))
        df = df.merge(m, on="MesRefCMO", how="left", validate="many_to_one")
        faltou = df[["PrecipitacaoMedia3M", "TemperaturaMedia3M", "CmoSeMesBase"]].isna().any(axis=1)
        exclusoes[f"t+{h}"] = {
            "AusenciaClimaONIBandeira": faltantes,
            "AusenciaCMOuJanela3M": df.loc[faltou, "MesAlvo"].astype(int).tolist()}
        bases[h] = df.loc[~faltou].copy()
    comuns = sorted(set.intersection(*(set(x["MesAlvo"]) for x in bases.values())))
    assert len(comuns) >= 84, "Base comum pequena demais para o protocolo."
    calendario = b["MesCompetencia"].tail(N_HOLDOUT).astype(int).tolist()
    for h in bases:
        df = bases[h]
        exclusoes[f"t+{h}"]["ForaIntersecaoHorizontes"] = (
            df.loc[~df["MesAlvo"].isin(comuns), "MesAlvo"].astype(int).tolist())
        df = df[df["MesAlvo"].isin(comuns)].sort_values("MesAlvo").reset_index(drop=True)
        exigir_continuidade(df["MesAlvo"])
        assert df["MesAlvo"].tail(N_HOLDOUT).tolist() == calendario, (
            "Features ausentes no teste; não deslocar calendário silenciosamente.")
        assert (df["MesRefCMO"] == df["MesReferenciaClima"]).all()
        assert np.isfinite(df.select_dtypes("number").to_numpy()).all()
        bases[h] = df
    return bases, calendario, exclusoes


def preparar_features_comuns(treino, avaliacao):
    normais = {}
    out_tr, out_va = treino.copy(), avaliacao.copy()
    for entrada, saida in [
        ("PrecipitacaoMediaMm", "PrecipitacaoPctNormalTreino"),
        ("PrecipitacaoMedia3M", "PrecipitacaoPctNormal3M"),
    ]:
        normal = treino.groupby("MesClima")[entrada].mean()
        assert len(normal) == 12 and (normal > 0).all()
        normais[entrada] = normal.to_dict()
        for out in [out_tr, out_va]:
            out[saida] = (100 * out[entrada] / out["MesClima"].map(normal)).round(1)
    return out_tr, out_va, normais


def dividir_desenvolvimento(df, h):
    teste = df.tail(N_HOLDOUT).copy()
    primeira_origem = int(teste["MesBase"].min())
    # Inclusive: a bandeira do mês da origem é conhecida por hipótese do projeto.
    treino = df[(df["MesAlvo"] < teste["MesAlvo"].min()) &
                (df["MesAlvo"] <= primeira_origem)].copy().reset_index(drop=True)
    excluidos = df[(df["MesAlvo"] < teste["MesAlvo"].min()) &
                   (df["MesAlvo"] > primeira_origem)]["MesAlvo"].astype(int).tolist()
    assert len(excluidos) == h - 1
    verificar_corte(treino, teste)
    splits = list(TimeSeriesSplit(
        n_splits=N_FOLDS, test_size=max(6, len(treino) // 10), gap=h-1).split(treino))
    return treino, teste, splits, excluidos


def probabilidades(modelo, X):
    p = np.asarray(modelo.predict_proba(X), dtype=float)
    assert list(modelo.classes_) == LABELS
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
    return p


def executar_comparacao(bases, calendario, exclusoes):
    inicio = time.perf_counter()
    resultados_cv, por_fold, auditoria, previsoes_cv = [], [], [], []
    resultados_teste, previsoes, tempos, selecao, comparacoes = [], [], [], [], []
    detalhes, modelos_finais = {}, {}
    for h, df in sorted(bases.items()):
        horizonte = f"t+{h}"
        dev, teste, splits, embargados = dividir_desenvolvimento(df, h)
        yh, pers_h = teste["NivelBandeira"].to_numpy(int), teste["NivelBandeiraMesBase"].to_numpy(int)
        assert teste["MesAlvo"].tolist() == calendario
        todos = REFERENCIAS + CANDIDATOS
        oof = {name: [] for name in todos}
        y_oof, registros_oof = [], []
        tempos_cv = {name: [] for name in CANDIDATOS}
        for i, (tr, va) in enumerate(splits, 1):
            treino, validacao = dev.iloc[tr], dev.iloc[va]
            verificar_corte(treino, validacao)
            assert set(treino["NivelBandeira"]) == set(LABELS)
            T, V, normas = preparar_features_comuns(treino, validacao)
            y = validacao["NivelBandeira"].to_numpy(int)
            preds = {
                "Persistencia": validacao["NivelBandeiraMesBase"].to_numpy(int),
                "ClasseMajoritaria": np.full(len(y), int(treino["NivelBandeira"].mode().iloc[0]))}
            for nome in CANDIDATOS:
                cols = colunas_candidato(nome, h)
                est = estimador_candidato(nome, h)
                ms = ajustar_cronometrado(est, T[cols], treino["NivelBandeira"])
                tempos_cv[nome].append(ms)
                preds[nome] = est.predict(V[cols]).astype(int)
            for nome in todos:
                oof[nome].extend(preds[nome].tolist())
                por_fold.append({"Horizonte": horizonte, "Fold": i, "Candidato": nome,
                                 "Nvalidacao": len(y), **avaliar(y, preds[nome])})
                for j, mes in enumerate(validacao["MesAlvo"]):
                    previsoes_cv.append({"Horizonte": horizonte, "Fold": i, "Candidato": nome,
                                         "MesAlvo": int(mes), "Real": int(y[j]),
                                         "Previsto": int(preds[nome][j])})
            y_oof.extend(y.tolist())
            auditoria.append({
                "Horizonte": horizonte, "Fold": i, "GapMeses": h-1,
                "TreinoInicio": int(treino["MesAlvo"].min()),
                "TreinoFim": int(treino["MesAlvo"].max()),
                "ValidacaoInicio": int(validacao["MesAlvo"].min()),
                "ValidacaoFim": int(validacao["MesAlvo"].max()),
                "PrimeiraOrigemValidacao": int(validacao["MesBase"].min()),
                "Ntreino": len(treino), "Nvalidacao": len(validacao)})
        linhas_cv = [{"Horizonte": horizonte, "Candidato": n, "NmesesOOF": len(y_oof),
                      **avaliar(y_oof, oof[n])} for n in todos]
        resultados_cv.extend(linhas_cv)
        def chave(r):
            return (-round(r["F1macro"], 12), r["MAEordinal"], PRIORIDADE[r["Candidato"]])
        ranking = sorted(linhas_cv, key=chave)
        selecionado = ranking[0]["Candidato"]
        melhor_ml = sorted([r for r in linhas_cv if r["Candidato"] in CANDIDATOS], key=chave)[0]
        # Seleção congelada antes de calcular QUALQUER métrica de holdout.
        selecao.append({"Horizonte": horizonte, "SelecionadoPelaCV": selecionado,
                        "F1macroCV": ranking[0]["F1macro"],
                        "MelhorAprendidoCV": melhor_ml["Candidato"],
                        "F1macroMelhorAprendidoCV": melhor_ml["F1macro"],
                        "TreinoFim": int(dev["MesAlvo"].max()),
                        "PrimeiraOrigemTeste": int(teste["MesBase"].min())})
        T, H, normas = preparar_features_comuns(dev, teste)
        previsoes_h = {
            "Persistencia": pers_h,
            "ClasseMajoritaria": np.full(len(yh), int(dev["NivelBandeira"].mode().iloc[0]))}
        proba_h, ajuste_h = {}, {}
        for nome in CANDIDATOS:
            cols = colunas_candidato(nome, h)
            est = estimador_candidato(nome, h)
            fit_ms = ajustar_cronometrado(est, T[cols], dev["NivelBandeira"])
            start_predict = time.perf_counter()
            previsoes_h[nome] = est.predict(H[cols]).astype(int)
            pred_ms = (time.perf_counter() - start_predict) * 1000
            proba_h[nome] = probabilidades(est, H[cols])
            modelos_finais[(h, nome)] = est
            tempos.append({"Horizonte": horizonte, "Candidato": nome,
                           "FitFinalMs": fit_ms, "FitCVTotalMs": float(sum(tempos_cv[nome])),
                           "PredictHoldoutMs": pred_ms,
                           "FitCVPorFoldMs": tempos_cv[nome]})
            ajuste_h[nome] = {"Features": cols, "FitConcluido": True}
            if nome == "Ordinal_CMO":
                ajuste_h[nome].update({
                    "CoeficientesPadronizados": dict(zip(cols, est.result_.params.iloc[:len(cols)].tolist())),
                    "LimiaresLatentes": est.model_.transform_threshold_params(est.result_.params)[1:-1].tolist(),
                    "Convergiu": bool(est.result_.mle_retvals["converged"])})
        for nome in todos:
            pred = previsoes_h[nome]
            linha = {"Horizonte": horizonte, "Candidato": nome,
                     "SelecionadoPelaCV": nome == selecionado,
                     "Ntreino": len(dev), "Nteste": len(teste),
                     "TreinoInicio": int(dev["MesAlvo"].min()),
                     "TreinoFim": int(dev["MesAlvo"].max()),
                     "TesteInicio": int(teste["MesAlvo"].min()),
                     "TesteFim": int(teste["MesAlvo"].max()), **avaliar(yh, pred)}
            if nome in proba_h:
                p = proba_h[nome]
                linha["LogLoss"] = float(log_loss(yh, p, labels=LABELS))
                linha["BrierMulticlasseSoma"] = float(np.mean(np.sum((p-np.eye(3)[yh])**2, axis=1)))
            resultados_teste.append(linha)
            for j, row in enumerate(teste.to_dict("records")):
                registro = {
                    "Horizonte": horizonte, "Candidato": nome,
                    **{k: int(row[k]) for k in ["MesBase", "MesAlvo", "MesReferenciaClima", "MesRefONI", "MesRefCMO"]},
                    "Real": int(yh[j]), "Previsto": int(pred[j]),
                    "BandeiraConhecida": int(pers_h[j]),
                    "SelecionadoPelaCV": nome == selecionado}
                if nome in proba_h:
                    registro.update(dict(zip(["ProbVerde", "ProbAmarela", "ProbVermelha"],
                                             proba_h[nome][j].tolist())))
                previsoes.append(registro)
            if nome in CANDIDATOS:
                comparacoes.append({"Horizonte": horizonte, "Candidato": nome,
                                    **comparar_persistencia(yh, pred, pers_h, h)})
        detalhes[horizonte] = {
            "BaseSHA256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
            "TotalBaseComum": len(df), "NormaisTreino": normas,
            "MesesEmbargadosAntesCV": embargados, "Exclusoes": exclusoes[horizonte],
            "Ajustes": ajuste_h,
            "MatrizesConfusao": {n: confusion_matrix(yh, previsoes_h[n], labels=LABELS).tolist() for n in todos}}
    payload = {
        "Titulo": "Comparação justa: v3, v5 e regressão ordinal",
        "ExecutadoUTC": datetime.now(timezone.utc).isoformat(),
        "Protocolo": "Receitas fixas; 3 classes; seleção por CV antes do teste; backtest revisitado",
        "Ambiente": {"Python": platform.python_version(), "Sklearn": sklearn.__version__,
                     "Statsmodels": statsmodels.__version__, "Scipy": scipy.__version__,
                     "Pandas": pd.__version__, "Numpy": np.__version__},
        "HoldoutMeses": calendario, "SelecaoCV": selecao, "ResumoCV": resultados_cv,
        "ResumoHoldout": resultados_teste, "MetricasPorFold": por_fold,
        "AuditoriaFolds": auditoria, "PrevisoesCV": previsoes_cv,
        "PrevisoesHoldout": previsoes, "Tempos": tempos,
        "ComparacaoPersistencia": comparacoes, "Detalhes": detalhes,
        "TempoAvaliacaoTotalMs": (time.perf_counter()-inicio)*1000}
    return payload, modelos_finais

# COMMAND ----------
# DBTITLE 1,Leitura do Unity Catalog e execução real, somente leitura
if "spark" in globals():
    from pyspark.sql import functions as F
    inicio_leitura = time.perf_counter()
    bandeira_pd = spark.table("mba.raw.bandeira_acionada").select(
        F.date_format("DatCompetencia", "yyyyMM").cast("int").alias("MesCompetencia"),
        "NomBandeiraAcionada").toPandas()
    clima_pd = spark.table("mba.refined.f_modelo_bandeira_clima").select(
        "MesCompetencia", "MesReferenciaClima", "PrecipitacaoMediaMm",
        "TemperaturaMediaC", "UmidadeMediaPct").toPandas()
    oni_pd = spark.table("mba.trusted.f_oni_enso").select("MesRef", "OniAnomC").toPandas()
    cmo_raw = spark.table("mba.trusted.f_cmo_subsistema").filter(F.col("id_subsistema") == "SE")
    assert cmo_raw.count() > 0, "CMO Sudeste vazio."
    assert cmo_raw.filter(F.col("din_instante").isNull() |
                          F.col("val_cmomediasemanal").isNull()).limit(1).count() == 0
    assert cmo_raw.groupBy("din_instante").count().filter("count > 1").limit(1).count() == 0, (
        "CMO Sudeste duplicado por data; revisar antes de agregar.")
    cmo_pd = cmo_raw.withColumn("MesRef", F.date_format("din_instante", "yyyyMM").cast("int")).groupBy(
        "MesRef").agg(F.avg("val_cmomediasemanal").alias("CmoSeMedioMensal")).toPandas()
    leitura_ms = (time.perf_counter() - inicio_leitura) * 1000
    inicio_preparo = time.perf_counter()
    bases, calendario, exclusoes = preparar_bases_comuns(bandeira_pd, clima_pd, oni_pd, cmo_pd)
    preparo_ms = (time.perf_counter() - inicio_preparo) * 1000
    resultado, modelos_finais = executar_comparacao(bases, calendario, exclusoes)
    resultado.update({"OrigemExecucao": "Databricks pessoal; Unity Catalog mba",
                      "TempoLeituraMs": leitura_ms, "TempoPreparoBaseMs": preparo_ms})
    print("A escolha vem da CV, não do maior número no teste já examinado.")
    for chave in ["SelecaoCV", "ResumoCV", "ResumoHoldout", "Tempos",
                  "ComparacaoPersistencia", "AuditoriaFolds", "PrevisoesHoldout"]:
        print(chave)
        display(pd.DataFrame(resultado[chave]))

# COMMAND ----------
# DBTITLE 1,Saída auditável, sem persistir tabelas nem alterar os notebooks anteriores
if "spark" in globals():
    dbutils.notebook.exit(json.dumps(resultado, ensure_ascii=False, allow_nan=False))
