# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Notebook 06: carga, ENA regional e evolução da EAR
# MAGIC
# MAGIC **Pergunta:** essas informações acrescentam poder preditivo à bandeira
# MAGIC conhecida e ao CMO, em t+1, t+2 e t+3?
# MAGIC
# MAGIC Este notebook é autônomo e SOMENTE LEITURA. Não altera tabelas, notebooks
# MAGIC anteriores, arquivos do repositório, Git ou agendamentos. Não usa `%run`.
# MAGIC Use **Import > File**, escolha este arquivo e depois **Run all**.
# MAGIC Precisa de acesso às cinco tabelas indicadas na célula de configuração.
# MAGIC Não cole tokens nem senhas. A leitura usa a sessão do Databricks.
# MAGIC
# MAGIC ## O que muda neste experimento
# MAGIC - Controle: regressão logística com bandeira conhecida e CMO Sudeste.
# MAGIC - Mais carga: controle + carga total do SIN, referência sazonal do treino
# MAGIC   e variação contra o mesmo mês do ano anterior.
# MAGIC - Mais ENA: controle + ENA bruta de N, NE, SE e S, cada uma relativa
# MAGIC   à referência sazonal aprendida SOMENTE no treino de cada janela.
# MAGIC - Mais EAR: controle + EAR percentual do último dia do mês de cada
# MAGIC   região e mudança acumulada em três meses, em pontos percentuais.
# MAGIC - Conjunto completo: controle + os três grupos.
# MAGIC - Referências concorrentes: persistência e classe majoritária do treino.
# MAGIC
# MAGIC É a MESMA regressão, regularização e regra de pesos nos cinco modelos.
# MAGIC Não há busca de hiperparâmetros, novas árvores ou escolha pelo teste.
# MAGIC O controle reproduz as entradas da receita v5 t+1, mas NÃO representa
# MAGIC as receitas climáticas de v5 t+2/t+3. Não depende de clima nem ONI.
# MAGIC
# MAGIC ## Relógio do experimento
# MAGIC A bandeira de t é conhecida. Carga, ENA, EAR e CMO terminam em t-1.
# MAGIC A variação de EAR é EAR(t-1) menos EAR(t-4). A variação anual de
# MAGIC carga compara t-1 com t-13. Nenhuma variável olha para t+1/t+2/t+3.
# MAGIC Os meses-alvo estão fixados em julho/2016 a agosto/2026, o mesmo
# MAGIC calendário do 05; teste fixado em setembro/2024 a agosto/2026.
# MAGIC Treino final: 98, 97 e 96 meses, respectivamente. Cinco janelas
# MAGIC expansivas de nove meses, gap=h-1. O embargo vem ANTES da seleção.
# MAGIC A escolha dos três horizontes é congelada pela CV antes do teste.
# MAGIC Scaler e referências sazonais nunca aprendem com validação ou teste.
# MAGIC No teste, o modelo fica congelado; as entradas e a bandeira já conhecida
# MAGIC são atualizadas a cada origem. Não são 24 meses previstos de uma só vez.
# MAGIC
# MAGIC ## Limites importantes
# MAGIC O teste já foi examinado: backtest revisitado, não confirmação inédita.
# MAGIC As fontes atuais não preservam todas as versões históricas publicadas.
# MAGIC Os lags são hipóteses de disponibilidade, não comprovação do dia de publicação.
# MAGIC A referência sazonal de ENA calculada aqui NÃO é a MLT oficial do ONS.
# MAGIC ENA bruta em MWmed representa energia afluente, não vazão em m³/s.
# MAGIC EAR é estoque, não geração; sua variação é em pontos percentuais.
# MAGIC Exigimos ao menos 95% dos dias válidos por mês e, para EAR, o último
# MAGIC dia do mês. Nenhum mês da amostra é removido ou preenchido silenciosamente.
# MAGIC Carga SIN é a média das somas diárias das quatro regiões completas;
# MAGIC essa agregação da série diária não é idêntica à série mensal oficial.
# MAGIC A composição da carga diária mudou em março/2021 e em 29/04/2023.
# MAGIC A auditoria sinaliza comparações anuais que atravessam esses regimes.
# MAGIC A referência sazonal da carga pode misturar regimes de medição.
# MAGIC Normalizar não corrige nem elimina essa quebra metodológica.
# MAGIC CMO mantém a média simples dos registros semanais pelo mês de sua data.
# MAGIC Audita todas as sextas-feiras, convenção das datas desta fonte semanal.
# MAGIC Uma mudança dessa convenção interrompe a execução para revisão.
# MAGIC Probabilidades não são calibradas. Coeficientes não demonstram causalidade.
# MAGIC Bootstrap em blocos é exploratório, sem ajuste por seleção/multiplicidade.
# MAGIC Tempos são medidos, não estimados; uma execução não é um benchmark.
# MAGIC Este é um experimento histórico, NÃO uma previsão operacional para hoje.
# MAGIC
# MAGIC Fontes e método:
# MAGIC - [ENA por subsistema, ONS](https://dados.ons.org.br/dataset/ena-diario-por-subsistema)
# MAGIC - [Composição da carga, ONS](https://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/carga_energia.aspx)
# MAGIC - [Validação temporal, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
# MAGIC
# COMMAND ----------
# DBTITLE 1,Configuração fixa: não ajustar parâmetros depois de olhar o teste
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, mean_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VERSAO = "06_carga_ena_ear_v1"
SEED = 42
REGIOES = ["N", "NE", "SE", "S"]
LABELS = [0, 1, 2]
NOMES_CLASSES = ["Verde", "Amarela", "Vermelha"]
MAPA_BANDEIRA = {
    "Verde": 0, "Amarela": 1, "Vermelha": 2, "Vermelha P1": 2,
    "Vermelha P2": 2, "Escassez Hídrica": 2, "Escassez Hidrica": 2,
}
INICIO_ALVOS = 201607
INICIO_TESTE = 202409
FIM_TESTE = 202608
INICIO_FONTES = 201401  # histórico de aquecimento; não aumenta o treino de rótulos
FIM_FONTES = 202606    # última entrada necessária: t+1 de agosto/2026 usa junho/2026
MIN_COBERTURA_DIAS = 0.95
N_FOLDS = 5
N_VALIDACAO = 9
N_BOOT = 2000
BLOCO_BOOT = 4

TABELAS = {
    "bandeira": "mba.raw.bandeira_acionada",
    "carga": "mba.trusted.f_carga_energia_subsistema",
    "ena": "mba.trusted.f_ena_subsistema",
    "ear": "mba.trusted.f_ear_subsistema",
    "cmo": "mba.trusted.f_cmo_subsistema",
}
# Cinco tabelas: bandeira, três fontes novas e o CMO usado como controle.
COLUNAS_BASE = ["NivelBandeiraMesBase", "CmoSeMesBase"]
COLUNAS_CARGA = ["CargaSIN_PctReferenciaTreino", "CargaSIN_Var12M_pct"]
COLUNAS_ENA = [f"ENA_{r}_PctReferenciaTreino" for r in REGIOES]
COLUNAS_EAR = (
    [f"EAR_{r}_FimMes_pct" for r in REGIOES]
    + [f"EAR_{r}_Var3M_pp" for r in REGIOES]
)
GRUPOS = {
    "Controle_CMO_Bandeira": COLUNAS_BASE,
    "Mais_Carga": COLUNAS_BASE + COLUNAS_CARGA,
    "Mais_ENA_Regional": COLUNAS_BASE + COLUNAS_ENA,
    "Mais_EAR_Evolucao": COLUNAS_BASE + COLUNAS_EAR,
    "Carga_ENA_EAR": COLUNAS_BASE + COLUNAS_CARGA + COLUNAS_ENA + COLUNAS_EAR,
}
REFERENCIAS = ["Persistencia", "ClasseMajoritaria"]
TODOS = REFERENCIAS + list(GRUPOS)
PRIORIDADE = {nome: i for i, nome in enumerate(TODOS)}

# COMMAND ----------
# DBTITLE 1,Datas e qualidade: falhar com mensagem em vez de esconder lacunas
def periodo(mes):
    s = str(int(mes))
    assert len(s) == 6 and 1 <= int(s[4:]) <= 12, f"Mês inválido: {mes}"
    return pd.Period(s[:4] + "-" + s[4:], freq="M")


def mes_int(p):
    return int(p.year * 100 + p.month)


def meses(inicio, fim):
    return [mes_int(p) for p in pd.period_range(periodo(inicio), periodo(fim), freq="M")]


def adicionar_meses(mes, n):
    return mes_int(periodo(mes) + n)


def exigir_continuidade(valores, nome):
    valores = list(map(int, valores))
    assert valores and valores == meses(valores[0], valores[-1]), f"{nome}: lacuna ou desordem mensal."


def preparar_bandeiras(entrada):
    b = entrada[["MesCompetencia", "NomBandeiraAcionada"]].copy()
    b["MesCompetencia"] = pd.to_numeric(b["MesCompetencia"], errors="raise").astype(int)
    assert not b["MesCompetencia"].duplicated().any(), "Bandeira duplicada por competência."
    b = b[b.MesCompetencia.between(201501, FIM_TESTE)].sort_values("MesCompetencia")
    assert b.NomBandeiraAcionada.notna().all(), "Bandeira nula."
    nomes = b.NomBandeiraAcionada.astype(str).str.strip()
    assert set(nomes) <= set(MAPA_BANDEIRA), f"Bandeira desconhecida: {set(nomes) - set(MAPA_BANDEIRA)}"
    b["NivelBandeira"] = nomes.map(MAPA_BANDEIRA).astype(int)
    exigir_continuidade(b.MesCompetencia, "Bandeiras")
    assert b.MesCompetencia.max() == FIM_TESTE, "Falta a última bandeira do teste fixo."
    return b.reset_index(drop=True)


def preparar_diario(entrada, nome, limite_superior=None):
    """Contrato comum: Data, Regiao, Valor. Mantém nulos para auditar cobertura."""
    df = entrada[["Data", "Regiao", "Valor"]].copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="raise").dt.normalize()
    assert df.Data.notna().all(), f"{nome}: data nula."
    inicio = periodo(INICIO_FONTES).start_time
    fim = periodo(FIM_FONTES).end_time.normalize()
    df = df[df.Data.between(inicio, fim)].copy()
    assert not df.empty, f"{nome}: vazio no período necessário."
    assert df.Regiao.notna().all(), f"{nome}: região nula."
    df["Regiao"] = df.Regiao.astype(str).str.strip().str.upper()
    assert set(df.Regiao) == set(REGIOES), f"{nome}: regiões inesperadas/incompletas."
    duplicados = df.duplicated(["Data", "Regiao"], keep=False)
    assert not duplicados.any(), (
        f"{nome}: duplicidade diária. Exemplos: "
        f"{df.loc[duplicados, ['Data', 'Regiao']].head(5).to_dict('records')}")
    df["Valor"] = pd.to_numeric(df["Valor"], errors="raise")
    validos = df.Valor.dropna().to_numpy(float)
    assert np.isfinite(validos).all() and (validos >= 0).all(), f"{nome}: valor negativo/infinito."
    if limite_superior is not None:
        assert (validos <= limite_superior + 1e-6).all(), f"{nome}: valor acima de {limite_superior}."
    # Reindexar o calendário ANTES de shift/rolling: dias e meses faltantes não desaparecem.
    indice = pd.date_range(inicio, fim, freq="D")
    return df.pivot(index="Data", columns="Regiao", values="Valor").reindex(
        index=indice, columns=REGIOES)


def agregar_serie(serie, fonte, regiao, usar_ultimo=False):
    chaves = serie.index.to_period("M")
    grupos = serie.groupby(chaves)
    contagem = grupos.count()
    dias = pd.Series([p.days_in_month for p in contagem.index], index=contagem.index)
    cobertura = contagem / dias
    if usar_ultimo:
        datas_fim = contagem.index.to_timestamp(how="end").normalize()
        valores = pd.Series(serie.reindex(datas_fim).to_numpy(), index=contagem.index)
    else:
        valores = grupos.mean()
    aprovados = (cobertura >= MIN_COBERTURA_DIAS) & valores.notna()
    qualidade = pd.DataFrame({
        "Fonte": fonte, "Regiao": regiao,
        "MesDados": [mes_int(p) for p in contagem.index],
        "DiasValidos": contagem.to_numpy(int), "DiasEsperados": dias.to_numpy(int),
        "Cobertura": cobertura.to_numpy(float),
        "UltimoDiaExigido": usar_ultimo, "Aprovado": aprovados.to_numpy(bool),
    })
    return valores.where(aprovados), qualidade


def regime_carga(p):
    # Abril/2023 contém dias antes e depois da mudança de 29/04.
    if p < periodo(202103):
        return "Supervisao"
    if p < periodo(202304):
        return "IncluiNaoDespachadas"
    if p == periodo(202304):
        return "Abril2023Misto"
    return "IncluiMMGD"


def preparar_energia(carga_entrada, ena_entrada, ear_entrada):
    carga = preparar_diario(carga_entrada, "Carga")
    ena = preparar_diario(ena_entrada, "ENA bruta")
    ear = preparar_diario(ear_entrada, "EAR percentual", limite_superior=100)
    indice = pd.period_range(periodo(INICIO_FONTES), periodo(FIM_FONTES), freq="M")
    mensal, auditorias = pd.DataFrame(index=indice), []
    # Só somar um dia quando TODAS as quatro regiões estiverem presentes.
    carga_sin = carga.sum(axis=1, min_count=len(REGIOES))
    mensal["CargaSIN_MWmed"], q = agregar_serie(carga_sin, "CargaSIN", "SIN")
    auditorias.append(q)
    # Também expor a cobertura de cada região; essas séries não são features de carga.
    for r in REGIOES:
        _, q = agregar_serie(carga[r], "CargaRegional", r)
        auditorias.append(q)
        mensal[f"ENA_{r}_MWmed"], q = agregar_serie(ena[r], "ENA", r)
        auditorias.append(q)
        mensal[f"EAR_{r}_FimMes_pct"], q = agregar_serie(ear[r], "EAR", r, usar_ultimo=True)
        auditorias.append(q)
        # Diferença entre pontos finais separados por três MESES, não três linhas válidas.
        mensal[f"EAR_{r}_Var3M_pp"] = (
            mensal[f"EAR_{r}_FimMes_pct"] - mensal[f"EAR_{r}_FimMes_pct"].shift(3))
    anterior = mensal.CargaSIN_MWmed.shift(12)
    mensal["CargaSIN_Var12M_pct"] = 100 * (mensal.CargaSIN_MWmed / anterior.where(anterior > 0) - 1)
    mensal["RegimeCarga"] = [regime_carga(p) for p in indice]
    mensal["CargaVar12M_CruzaRegime"] = [
        regime_carga(p) != regime_carga(p - 12) for p in indice]
    mensal["MesDados"] = [mes_int(p) for p in indice]
    mensal["MesSazonal"] = [p.month for p in indice]
    return mensal.reset_index(drop=True), pd.concat(auditorias, ignore_index=True)


def agregar_cmo_semanal(entrada):
    """Datas da fonte CMO são sextas-feiras; não confundir com data de publicação."""
    c = entrada[["Data", "Valor"]].copy()
    c["Data"] = pd.to_datetime(c.Data, errors="raise").dt.normalize()
    assert c.Data.notna().all(), "CMO com data nula."
    inicio, fim = periodo(INICIO_FONTES).start_time, periodo(FIM_FONTES).end_time.normalize()
    c = c[c.Data.between(inicio, fim)].copy()
    assert not c.empty and not c.Data.duplicated().any(), "CMO semanal vazio/duplicado."
    assert (c.Data.dt.dayofweek == 4).all(), "Convenção de datas do CMO mudou; revisar antes de agregar."
    c["Valor"] = pd.to_numeric(c.Valor, errors="raise")
    validos = c.Valor.dropna().to_numpy(float)
    assert np.isfinite(validos).all() and (validos >= 0).all(), "CMO inválido."
    esperadas = pd.date_range(inicio, fim, freq="W-FRI")
    serie = c.set_index("Data").Valor.reindex(esperadas)
    grupos = serie.groupby(serie.index.to_period("M"))
    n_validas, n_esperadas = grupos.count(), grupos.size()
    completo = n_validas == n_esperadas
    return pd.DataFrame({
        "MesRef": [mes_int(p) for p in n_validas.index],
        "CmoSeMedioMensal": grupos.mean().where(completo).to_numpy(),
        "SemanasValidas": n_validas.to_numpy(int),
        "SemanasEsperadas": n_esperadas.to_numpy(int),
        "MesCMOCompleto": completo.to_numpy(bool),
    })


def preparar_cmo(entrada):
    colunas = ["MesRef", "CmoSeMedioMensal", "SemanasValidas", "SemanasEsperadas", "MesCMOCompleto"]
    c = entrada[colunas].copy()
    c["MesRef"] = pd.to_numeric(c.MesRef, errors="raise").astype(int)
    c = c[c.MesRef.between(INICIO_FONTES, FIM_FONTES)].copy()
    assert not c.MesRef.duplicated().any(), "CMO duplicado por mês."
    c["CmoSeMedioMensal"] = pd.to_numeric(c.CmoSeMedioMensal, errors="raise")
    assert c.MesCMOCompleto.all() and (c.SemanasValidas == c.SemanasEsperadas).all(), (
        "CMO com semanas faltantes. Conferir qualidade_cmo antes de prosseguir.")
    assert np.isfinite(c.CmoSeMedioMensal.to_numpy(float)).all(), "CMO nulo/infinito."
    assert (c.CmoSeMedioMensal >= 0).all(), "CMO negativo."
    return c.rename(columns={"MesRef": "MesDados", "CmoSeMedioMensal": "CmoSeMesBase"})


def montar_bases(bandeiras, energia, cmo):
    b, m = preparar_bandeiras(bandeiras), preparar_cmo(cmo)
    assert not energia.MesDados.duplicated().any(), "Energia duplicada por mês."
    respostas = b[["MesCompetencia", "NivelBandeira"]]
    entradas = ["CargaSIN_MWmed", "CargaSIN_Var12M_pct", "CmoSeMesBase"] + (
        [f"ENA_{r}_MWmed" for r in REGIOES] + COLUNAS_EAR)
    bases = {}
    for h in [1, 2, 3]:
        df = pd.DataFrame({"MesAlvo": meses(INICIO_ALVOS, FIM_TESTE)})
        df["MesBase"] = df.MesAlvo.map(lambda x: adicionar_meses(x, -h))
        df["MesDados"] = df.MesBase.map(lambda x: adicionar_meses(x, -1))
        df["MesEARComparacao"] = df.MesDados.map(lambda x: adicionar_meses(x, -3))
        df["MesCargaComparacao"] = df.MesDados.map(lambda x: adicionar_meses(x, -12))
        df = df.merge(respostas.rename(columns={"MesCompetencia": "MesAlvo"}),
                      on="MesAlvo", how="left", validate="one_to_one")
        df = df.merge(respostas.rename(columns={
            "MesCompetencia": "MesBase", "NivelBandeira": "NivelBandeiraMesBase"}),
            on="MesBase", how="left", validate="one_to_one")
        df = df.merge(energia, on="MesDados", how="left", validate="one_to_one")
        df = df.merge(m, on="MesDados", how="left", validate="one_to_one")
        obrigatorias = entradas + ["NivelBandeira", "NivelBandeiraMesBase", "MesSazonal"]
        faltou = df[obrigatorias].isna().any(axis=1)
        assert not faltou.any(), (
            f"t+{h}: faltam entradas/qualidade nos alvos {df.loc[faltou, 'MesAlvo'].tolist()}. "
            "Consultar qualidade_mensal. Corrigir a fonte, não deslocar o teste ou remover linhas.")
        assert np.isfinite(df[obrigatorias].to_numpy(float)).all(), f"t+{h}: entrada não finita."
        for col in ["NivelBandeira", "NivelBandeiraMesBase", "MesSazonal"]:
            df[col] = df[col].astype(int)
        assert (df.MesDados < df.MesBase).all() and (df.MesBase < df.MesAlvo).all()
        assert df.MesAlvo.tolist() == meses(INICIO_ALVOS, FIM_TESTE)
        bases[h] = df
    return bases

# COMMAND ----------
# DBTITLE 1,Referências sazonais e escalonamento aprendidos somente no treino
def preparar_features(treino, avaliacao):
    T, V, referencias = treino.copy(), avaliacao.copy(), []
    pares = [("CargaSIN_MWmed", "CargaSIN_PctReferenciaTreino")] + [
        (f"ENA_{r}_MWmed", f"ENA_{r}_PctReferenciaTreino") for r in REGIOES]
    for origem, destino in pares:
        medias = treino.groupby("MesSazonal")[origem].mean()
        contagens = treino.groupby("MesSazonal")[origem].count()
        assert set(medias.index) == set(range(1, 13)), "Treino sem os 12 meses sazonais."
        assert (medias > 0).all() and (contagens >= 2).all(), "Referência sazonal insuficiente."
        for out in [T, V]:
            out[destino] = 100 * out[origem] / out.MesSazonal.map(medias)
        referencias.extend({
            "Variavel": origem, "MesSazonal": int(k),
            "MediaTreino": float(medias[k]), "Nobservacoes": int(contagens[k]),
        } for k in medias.index)
    colunas = GRUPOS["Carga_ENA_EAR"]
    assert np.isfinite(T[colunas].to_numpy(float)).all()
    assert np.isfinite(V[colunas].to_numpy(float)).all()
    return T, V, referencias


def criar_modelo():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs",
            max_iter=2000, random_state=SEED)),
    ])


def ajustar(modelo, X, y):
    assert set(y) == set(LABELS), "Treino sem as três classes: não omitir o fold."
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        inicio = time.perf_counter()
        modelo.fit(X, y)
        return (time.perf_counter() - inicio) * 1000


def verificar_corte(treino, avaliacao):
    assert treino.MesAlvo.max() < avaliacao.MesAlvo.min()
    assert treino.MesAlvo.max() <= avaliacao.MesBase.min(), (
        "Rótulo de treino posterior à primeira origem de avaliação.")


def dividir_desenvolvimento(df, h):
    teste = df[df.MesAlvo.between(INICIO_TESTE, FIM_TESTE)].copy()
    assert teste.MesAlvo.tolist() == meses(INICIO_TESTE, FIM_TESTE)
    dev = df[(df.MesAlvo < INICIO_TESTE) & (df.MesAlvo <= teste.MesBase.min())].copy()
    embargo = df[(df.MesAlvo < INICIO_TESTE) & (df.MesAlvo > teste.MesBase.min())]
    assert len(embargo) == h - 1
    assert len(dev) == 99 - h, "O calendário não corresponde ao protocolo fixado."
    exigir_continuidade(dev.MesAlvo, "Desenvolvimento")
    verificar_corte(dev, teste)
    splits = list(TimeSeriesSplit(
        n_splits=N_FOLDS, test_size=N_VALIDACAO, gap=h-1).split(dev))
    return dev.reset_index(drop=True), teste.reset_index(drop=True), splits, embargo.MesAlvo.tolist()


def avaliar(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    f1 = f1_score(y, pred, labels=LABELS, average=None, zero_division=0)
    kappa = (float(cohen_kappa_score(y, pred, labels=LABELS, weights="quadratic"))
             if len(set(y) | set(pred)) > 1 else None)
    return {
        "F1macro": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "Acuracia": float(accuracy_score(y, pred)),
        "MAEordinal": float(mean_absolute_error(y, pred)),
        "KappaQuadratico": kappa,
        **{f"F1{nome}": float(valor) for nome, valor in zip(NOMES_CLASSES, f1)},
    }


def comparar_blocos(y, pred, persistencia, h):
    """Diferença pareada de F1; IC exploratório, sem ajuste da seleção."""
    y, pred, persistencia = map(lambda x: np.asarray(x, int), [y, pred, persistencia])
    def f1(a, b):
        cm = np.bincount(a * 3 + b, minlength=9).reshape(3, 3)
        den = cm.sum(0) + cm.sum(1)
        return float(np.divide(2 * cm.diagonal(), den, out=np.zeros(3), where=den != 0).mean())
    rng, n = np.random.default_rng(SEED + h), len(y)
    deltas = []
    for _ in range(N_BOOT):
        inicios = rng.integers(0, n, size=int(np.ceil(n / BLOCO_BOOT)))
        idx = ((inicios[:, None] + np.arange(BLOCO_BOOT)) % n).ravel()[:n]
        deltas.append(f1(y[idx], pred[idx]) - f1(y[idx], persistencia[idx]))
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "DeltaF1VsPersistencia": f1(y, pred) - f1(y, persistencia),
        "IC95ExploratorioInferior": float(low), "IC95ExploratorioSuperior": float(high),
        "SoModeloAcerta": int(((pred == y) & (persistencia != y)).sum()),
        "SoPersistenciaAcerta": int(((pred != y) & (persistencia == y)).sum()),
        "BlocoMeses": BLOCO_BOOT, "Repeticoes": N_BOOT,
    }


def linha_metricas(h, nome, y, pred):
    return {"Horizonte": f"t+{h}", "Candidato": nome,
            "Nmeses": len(y), "Nfeatures": len(GRUPOS.get(nome, [])), **avaliar(y, pred)}


def chave_selecao(row):
    # Primeiro o critério predefinido; simplicidade só desempata.
    return (-round(row["F1macro"], 12), row["MAEordinal"],
            row["Nfeatures"], PRIORIDADE[row["Candidato"]])


def auditar_regime_carga(df, h, etapa):
    return {
        "Horizonte": f"t+{h}", "Etapa": etapa, "Nmeses": len(df),
        "NcomparacoesAnuaisEntreRegimes": int(df.CargaVar12M_CruzaRegime.sum()),
        "MesesDadosComparadosEntreRegimes": df.loc[
            df.CargaVar12M_CruzaRegime, "MesDados"].astype(int).tolist(),
        "RegimesPresentes": sorted(df.RegimeCarga.unique().tolist()),
    }


def executar_experimento(bases):
    inicio = time.perf_counter()
    resumo_cv, folds, auditoria, previsoes_cv, referencias = [], [], [], [], []
    selecao, contextos, fit_cv, regimes = [], {}, {}, []
    # FASE A: concluir TODA a CV e congelar as três escolhas antes do holdout.
    for h, df in sorted(bases.items()):
        dev, teste, splits, embargo = dividir_desenvolvimento(df, h)
        contextos[h] = (dev, teste, embargo)
        oof = {nome: [] for nome in TODOS}
        y_oof = []
        for nome in GRUPOS:
            fit_cv[(h, nome)] = []
        for k, (tr, va) in enumerate(splits, 1):
            treino, validacao = dev.iloc[tr], dev.iloc[va]
            verificar_corte(treino, validacao)
            regimes.extend([auditar_regime_carga(treino, h, f"CV{k}_Treino"),
                            auditar_regime_carga(validacao, h, f"CV{k}_Validacao")])
            T, V, ref = preparar_features(treino, validacao)
            referencias.extend({"Horizonte": f"t+{h}", "Etapa": f"CV{k}", **x} for x in ref)
            y = validacao.NivelBandeira.to_numpy(int)
            preds = {
                "Persistencia": validacao.NivelBandeiraMesBase.to_numpy(int),
                "ClasseMajoritaria": np.full(len(y), int(treino.NivelBandeira.mode().iloc[0])),
            }
            for nome, cols in GRUPOS.items():
                est = criar_modelo()
                fit_cv[(h, nome)].append(ajustar(est, T[cols], treino.NivelBandeira))
                preds[nome] = est.predict(V[cols]).astype(int)
            for nome in TODOS:
                oof[nome].extend(preds[nome].tolist())
                folds.append({"Fold": k, **linha_metricas(h, nome, y, preds[nome])})
                for j, row in enumerate(validacao.itertuples()):
                    previsoes_cv.append({
                        "Horizonte": f"t+{h}", "Fold": k, "Candidato": nome,
                        "MesBase": int(row.MesBase), "MesAlvo": int(row.MesAlvo),
                        "MesDados": int(row.MesDados), "Real": int(y[j]),
                        "Previsto": int(preds[nome][j]),
                        "BandeiraConhecida": int(row.NivelBandeiraMesBase),
                    })
            y_oof.extend(y.tolist())
            auditoria.append({
                "Horizonte": f"t+{h}", "Fold": k, "GapMeses": h-1,
                "TreinoInicio": int(treino.MesAlvo.min()), "TreinoFim": int(treino.MesAlvo.max()),
                "ValidacaoInicio": int(validacao.MesAlvo.min()),
                "ValidacaoFim": int(validacao.MesAlvo.max()),
                "PrimeiraOrigemValidacao": int(validacao.MesBase.min()),
                "UltimoMesDadosTreino": int(treino.MesDados.max()),
                "Ntreino": len(treino), "Nvalidacao": len(validacao),
            })
        linhas = [linha_metricas(h, nome, y_oof, oof[nome]) for nome in TODOS]
        resumo_cv.extend(linhas)
        escolhido = min(linhas, key=chave_selecao)
        aprendido = min([r for r in linhas if r["Candidato"] in GRUPOS], key=chave_selecao)
        controle = next(r for r in linhas if r["Candidato"] == "Controle_CMO_Bandeira")
        pers = next(r for r in linhas if r["Candidato"] == "Persistencia")
        selecao.append({
            "Horizonte": f"t+{h}", "SelecionadoPelaCV": escolhido["Candidato"],
            "F1macroCV": escolhido["F1macro"], "MelhorAprendidoCV": aprendido["Candidato"],
            "F1macroMelhorAprendidoCV": aprendido["F1macro"],
            "DeltaCV_MelhorAprendidoVsControle": aprendido["F1macro"] - controle["F1macro"],
            "DeltaCV_MelhorAprendidoVsPersistencia": aprendido["F1macro"] - pers["F1macro"],
            "NtreinoFinal": len(dev), "TreinoFinalFim": int(dev.MesAlvo.max()),
            "EmbargoAntesSelecao": list(map(int, embargo)),
        })
    escolhas_congeladas = {x["Horizonte"]: x["SelecionadoPelaCV"] for x in selecao}

    # FASE B: avaliar o teste fixo. Os rótulos de teste não entram na seleção.
    resumo_teste, previsoes, tempos, matrizes, transicoes = [], [], [], [], []
    comparacoes, coeficientes, modelos_finais, features_teste = [], [], {}, []
    for h, (dev, teste, embargo) in sorted(contextos.items()):
        regimes.extend([auditar_regime_carga(dev, h, "TreinoFinal"),
                        auditar_regime_carga(teste, h, "Teste")])
        T, H, ref = preparar_features(dev, teste)
        referencias.extend({"Horizonte": f"t+{h}", "Etapa": "FitFinal", **x} for x in ref)
        y, pers = teste.NivelBandeira.to_numpy(int), teste.NivelBandeiraMesBase.to_numpy(int)
        preds = {"Persistencia": pers,
                 "ClasseMajoritaria": np.full(len(y), int(dev.NivelBandeira.mode().iloc[0]))}
        probabilidades = {}
        for nome, cols in GRUPOS.items():
            est = criar_modelo()
            fit_ms = ajustar(est, T[cols], dev.NivelBandeira)
            ini = time.perf_counter()
            preds[nome] = est.predict(H[cols]).astype(int)
            pred_ms = (time.perf_counter() - ini) * 1000
            p = np.asarray(est.predict_proba(H[cols]), float)
            assert list(est.classes_) == LABELS and p.shape == (len(H), 3)
            assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1)
            probabilidades[nome] = p
            modelos_finais[(h, nome)] = est
            tempos.append({
                "Horizonte": f"t+{h}", "Candidato": nome, "FitFinalMs": float(fit_ms),
                "FitCVTotalMs": float(sum(fit_cv[(h, nome)])),
                "PredictHoldoutMs": float(pred_ms),
                "FitCVPorFoldMs": list(map(float, fit_cv[(h, nome)])),
            })
            for classe, pesos in zip(est.classes_, est.named_steps["clf"].coef_):
                coeficientes.extend({
                    "Horizonte": f"t+{h}", "Candidato": nome,
                    "Classe": NOMES_CLASSES[int(classe)], "Variavel": col,
                    "CoeficientePadronizado": float(peso),
                } for col, peso in zip(cols, pesos))
        for nome in TODOS:
            pred = preds[nome]
            escolhido = escolhas_congeladas[f"t+{h}"] == nome
            row = {
                **linha_metricas(h, nome, y, pred), "SelecionadoPelaCV": escolhido,
                "Ntreino": len(dev), "TreinoInicio": int(dev.MesAlvo.min()),
                "TreinoFim": int(dev.MesAlvo.max()), "TesteInicio": INICIO_TESTE,
                "TesteFim": FIM_TESTE,
            }
            if nome in probabilidades:
                row["LogLossNaoCalibrada"] = float(log_loss(y, probabilidades[nome], labels=LABELS))
                comparacoes.append({"Horizonte": f"t+{h}", "Candidato": nome,
                                    **comparar_blocos(y, pred, pers, h)})
            resumo_teste.append(row)
            cm = confusion_matrix(y, pred, labels=LABELS)
            for a in LABELS:
                for b in LABELS:
                    matrizes.append({"Horizonte": f"t+{h}", "Candidato": nome,
                                     "Real": NOMES_CLASSES[a], "Previsto": NOMES_CLASSES[b],
                                     "Contagem": int(cm[a, b])})
            for descricao, mascara in [("MudouEntreOrigemEAlvo", y != pers),
                                       ("MesmaCorNaOrigemEAlvo", y == pers)]:
                transicoes.append({
                    "Horizonte": f"t+{h}", "Candidato": nome, "Situacao": descricao,
                    "Nmeses": int(mascara.sum()),
                    "Acuracia": float((pred[mascara] == y[mascara]).mean()) if mascara.any() else None,
                })
            for j, original in enumerate(teste.to_dict("records")):
                registro = {
                    "Horizonte": f"t+{h}", "Candidato": nome, "SelecionadoPelaCV": escolhido,
                    **{col: int(original[col]) for col in [
                        "MesAlvo", "MesBase", "MesDados", "MesEARComparacao", "MesCargaComparacao"]},
                    "BandeiraConhecida": int(pers[j]), "Real": int(y[j]), "Previsto": int(pred[j]),
                    "RealNome": NOMES_CLASSES[int(y[j])], "PrevistoNome": NOMES_CLASSES[int(pred[j])],
                    "Acertou": bool(y[j] == pred[j]),
                }
                if nome in probabilidades:
                    registro.update(dict(zip(
                        ["ProbVerde", "ProbAmarela", "ProbVermelha"],
                        probabilidades[nome][j].tolist())))
                previsoes.append(registro)
        cols_auditoria = ["MesAlvo", "MesBase", "MesDados", "RegimeCarga",
                          "CargaVar12M_CruzaRegime"] + GRUPOS["Carga_ENA_EAR"]
        features_teste.extend({"Horizonte": f"t+{h}", **x}
                              for x in H[cols_auditoria].to_dict("records"))
    assert escolhas_congeladas == {x["Horizonte"]: x["SelecionadoPelaCV"] for x in selecao}
    ganhos = []
    for h in [1, 2, 3]:
        linhas = {x["Candidato"]: x for x in resumo_cv if x["Horizonte"] == f"t+{h}"}
        for nome in GRUPOS:
            ganhos.append({
                "Horizonte": f"t+{h}", "Candidato": nome, "Nfeatures": len(GRUPOS[nome]),
                "F1macroCV": linhas[nome]["F1macro"],
                "DeltaF1CV_VsControle": linhas[nome]["F1macro"] - linhas["Controle_CMO_Bandeira"]["F1macro"],
                "DeltaF1CV_VsPersistencia": linhas[nome]["F1macro"] - linhas["Persistencia"]["F1macro"],
            })
    hashes = {f"t+{h}": hashlib.sha256(
        df.to_csv(index=False, float_format="%.12g").encode()).hexdigest()
        for h, df in bases.items()}
    resultado = {
        "Versao": VERSAO, "ExecucaoUTC": datetime.now(timezone.utc).isoformat(),
        "Ambiente": {"Python": platform.python_version(), "Pandas": pd.__version__,
                     "Numpy": np.__version__, "Sklearn": sklearn.__version__},
        "Protocolo": {
            "InicioAlvos": INICIO_ALVOS, "InicioTeste": INICIO_TESTE, "FimTeste": FIM_TESTE,
            "NFolds": N_FOLDS, "NValidacaoPorFold": N_VALIDACAO, "Gap": "h-1",
            "Selecao": "F1macro OOF concatenado (45 meses), nao media dos F1 dos folds",
            "Desempate": "F1 arredondado a 12 casas desc; MAE asc; numero de features asc; prioridade fixa",
            "PrioridadeFixa": TODOS, "LabelsFixos": LABELS, "ZeroDivision": 0,
            "ModaEmpatada": "Menor classe numerica do treino",
            "DadosEm": "t-1", "CargaVar12M": "(carga(t-1)/carga(t-13)-1)*100",
            "EARVar3M": "EAR(t-1)-EAR(t-4), pontos percentuais",
            "NormalENA": "Media do mesmo mes calculada exclusivamente no treino; nao e MLT oficial",
            "MinCoberturaDiaria": MIN_COBERTURA_DIAS,
            "Modelo": "StandardScaler + LogisticRegression(C=1, class_weight=balanced, lbfgs)",
            "BacktestRevisitado": True, "VintagesHistoricosDisponiveis": False,
            "ParametrosFixadosAntesDaExecucao": True, "EscolhaUsaHoldout": False,
        },
        "FeaturesPorCandidato": GRUPOS, "SHA256Bases": hashes,
        "BasesMensais": [{"Horizonte": f"t+{h}", **row}
                         for h, df in sorted(bases.items()) for row in df.to_dict("records")],
        "RegimesCarga": regimes,
        "SelecaoCV": selecao, "ResumoCV": resumo_cv, "GanhoIncrementalCV": ganhos,
        "MetricasPorFold": folds, "AuditoriaFolds": auditoria,
        "PrevisoesCV": previsoes_cv, "ResumoHoldout": resumo_teste,
        "PrevisoesHoldout": previsoes, "MatrizesConfusao": matrizes,
        "DiagnosticoMudancas": transicoes, "ComparacaoPersistencia": comparacoes,
        "Tempos": tempos, "Coeficientes": coeficientes,
        "ReferenciasSazonais": referencias, "FeaturesTeste": features_teste,
        "TempoExperimentoMs": (time.perf_counter() - inicio) * 1000,
    }
    # Recusar saída com NaN/Infinity, inclusive em métricas e auditorias.
    json.dumps(resultado, ensure_ascii=False, allow_nan=False)
    return resultado, modelos_finais

# COMMAND ----------
# DBTITLE 1,Ler o Unity Catalog: nenhum comando de escrita ou alteração
def ler_databricks(spark):
    from pyspark.sql import functions as F
    for nome in TABELAS.values():
        assert spark.catalog.tableExists(nome), (
            f"Tabela {nome} não encontrada/acessível. Validar a ingestão e permissões.")
    b = spark.table(TABELAS["bandeira"]).select(
        F.date_format("DatCompetencia", "yyyyMM").cast("int").alias("MesCompetencia"),
        "NomBandeiraAcionada").toPandas()
    def diario(tabela, data, valor):
        raw = spark.table(tabela).select(
            F.to_date(F.col(data)).alias("Data"),
            F.col("id_subsistema").alias("Regiao"), F.col(valor).alias("Valor")
        )
        assert raw.filter(F.col("Data").isNull()).limit(1).count() == 0, (
            f"{tabela}: data nula/inválida, impossível atribuir o registro a um mês.")
        return raw.filter(
            (F.col("Data") >= F.lit("2014-01-01").cast("date")) &
            (F.col("Data") <= F.lit("2026-06-30").cast("date"))
        ).toPandas()
    carga = diario(TABELAS["carga"], "din_instante", "val_cargaenergiamwmed")
    ena = diario(TABELAS["ena"], "ena_data", "ena_bruta_regiao_mwmed")
    ear = diario(TABELAS["ear"], "ear_data", "ear_verif_subsistema_percentual")
    raw = spark.table(TABELAS["cmo"]).filter(F.col("id_subsistema") == "SE").select(
        "din_instante", "val_cmomediasemanal")
    assert raw.filter(F.col("din_instante").isNull()).limit(1).count() == 0, "CMO com data nula."
    raw = raw.filter(
            (F.col("din_instante") >= F.lit("2014-01-01").cast("date")) &
            (F.col("din_instante") <= F.lit("2026-06-30").cast("date")))
    cmo = agregar_cmo_semanal(raw.select(
        F.col("din_instante").alias("Data"), F.col("val_cmomediasemanal").alias("Valor")
    ).toPandas())
    return b, carga, ena, ear, cmo


if "spark" in globals():
    inicio_leitura = time.perf_counter()
    bandeiras_pd, carga_pd, ena_pd, ear_pd, cmo_pd = ler_databricks(spark)
    tempo_leitura_ms = (time.perf_counter() - inicio_leitura) * 1000
    inicio_preparo = time.perf_counter()
    energia_mensal, qualidade_mensal = preparar_energia(carga_pd, ena_pd, ear_pd)
    print("Qualidade das fontes: meses com falha ou com menos de 100% dos dias válidos.")
    display(qualidade_mensal[
        (~qualidade_mensal.Aprovado) | (qualidade_mensal.Cobertura < 1)])
    qualidade_cmo = cmo_pd.copy()
    print("Qualidade do CMO: meses com semanas faltantes; tabela vazia é o esperado.")
    display(qualidade_cmo[~qualidade_cmo.MesCMOCompleto])
    bases = montar_bases(bandeiras_pd, energia_mensal, cmo_pd)
    tempo_preparo_ms = (time.perf_counter() - inicio_preparo) * 1000
    resultado, modelos_finais = executar_experimento(bases)
    resultado.update({
        "OrigemExecucao": "Databricks; tabelas do Unity Catalog",
        "TempoLeituraMs": tempo_leitura_ms, "TempoPreparoMs": tempo_preparo_ms,
        "TabelasConsultadas": TABELAS, "QualidadeMensal": qualidade_mensal.to_dict("records"),
        "QualidadeCMO": qualidade_cmo.to_dict("records"),
    })

# COMMAND ----------
# DBTITLE 1,Resultados: ler primeiro a seleção e o ganho na validação
if "spark" in globals():
    selecao_cv = pd.DataFrame(resultado["SelecaoCV"])
    resumo_cv = pd.DataFrame(resultado["ResumoCV"])
    ganho_incremental_cv = pd.DataFrame(resultado["GanhoIncrementalCV"])
    resumo_teste = pd.DataFrame(resultado["ResumoHoldout"])
    real_x_previsto = pd.DataFrame(resultado["PrevisoesHoldout"])
    auditoria_folds = pd.DataFrame(resultado["AuditoriaFolds"])
    bases_mensais = pd.DataFrame(resultado["BasesMensais"])
    regimes_carga = pd.DataFrame(resultado["RegimesCarga"])
    tempos = pd.DataFrame(resultado["Tempos"])
    print("Escolhas fixadas pela CV. Não escolher novamente pelo teste já examinado.")
    for titulo, tabela in [
        ("Seleção pela CV", selecao_cv), ("Comparação dos grupos na CV", ganho_incremental_cv),
        ("Métricas completas da CV", resumo_cv), ("Teste histórico revisitado", resumo_teste),
        ("Cortes temporais", auditoria_folds), ("Tempos reais desta execução", tempos),
        ("Regimes de medição da carga, sem harmonização", regimes_carga),
        ("Bases mensais usadas: preserve para reprodução", bases_mensais),
        ("Real versus previsto", real_x_previsto),
        ("Acertos em mudanças e permanências", pd.DataFrame(resultado["DiagnosticoMudancas"])),
        ("Diferenças versus persistência: IC exploratório", pd.DataFrame(resultado["ComparacaoPersistencia"])),
        ("Matrizes de confusão", pd.DataFrame(resultado["MatrizesConfusao"])),
    ]:
        print(titulo)
        display(tabela)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Como interpretar
# MAGIC 1. Veja `selecao_cv`: se aparecer Persistencia, a regra simples venceu
# MAGIC    novamente neste protocolo. Não é motivo para esconder o resultado.
# MAGIC 2. Veja `ganho_incremental_cv`: delta positivo versus controle significa
# MAGIC    ganho ao acrescentar o grupo; delta negativo indica piora.
# MAGIC    O F1 é calculado nos 45 meses OOF concatenados, não pela média dos folds.
# MAGIC    Empates usam menor MAE, menos entradas e a prioridade fixa do código.
# MAGIC 3. Veja `resumo_teste` apenas como diagnóstico histórico revisitado.
# MAGIC 4. Veja `real_x_previsto`, datas de origem/alvo e erros de cada classe.
# MAGIC 5. A situação "mesma cor" compara os pontos origem/alvo: em h>1 pode
# MAGIC    ter havido mudança e retorno entre eles. Não significa trajetória constante.
# MAGIC 6. `tempos` separa fit de leitura, preparo e previsão. Não compara
# MAGIC    hardware de forma controlada nem garante desempenho operacional.
# MAGIC 7. As tabelas são exibidas, mas não salvas no catálogo. Use a opção de
# MAGIC    download da tabela, se disponível, para guardar os CSVs da sua execução.
# MAGIC 8. Não altere parâmetros repetidamente para perseguir o melhor teste.
# MAGIC    Uma futura avaliação independente precisa de meses ainda não examinados.
# MAGIC
# COMMAND ----------
# DBTITLE 1,Registro JSON para auditoria e execuções como job
if "spark" in globals():
    resultado_json = json.dumps(resultado, ensure_ascii=False, allow_nan=False)
    print(f"Registro JSON preparado: {len(resultado_json.encode('utf-8')):,} bytes.")
    # Devolve o resultado da execução sem escrever arquivos/tabelas.
    dbutils.notebook.exit(resultado_json)
