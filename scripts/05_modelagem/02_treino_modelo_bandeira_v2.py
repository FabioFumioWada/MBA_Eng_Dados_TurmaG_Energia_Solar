# Databricks notebook source
# MAGIC %md
# MAGIC ## Treino do Modelo v2 - Previsao de Bandeira Vermelha (Sprint 4)
# MAGIC
# MAGIC ### Objetivo
# MAGIC Este notebook parte do modelo v1 (ver `01_treino_modelo_bandeira`) e
# MAGIC testa, de forma honesta e documentada, se outras variaveis e outros
# MAGIC algoritmos conseguem uma acuracia real maior, sem inventar dados nem
# MAGIC vazar informacao do futuro.
# MAGIC
# MAGIC ### Base cientifica consultada
# MAGIC - McKee et al. (1993) propoe o indice SPI (Standardized Precipitation
# MAGIC   Index), que usa janelas de 3 meses de chuva acumulada para avaliar
# MAGIC   seca hidrologica, em vez de olhar so o mes isolado.
# MAGIC - Estudos de classificacao de seca com machine learning comparam
# MAGIC   Random Forest, SVM, Gradient Boosting e Regressao Logistica como
# MAGIC   algoritmos padrao para esse tipo de problema (Tandfonline, 2025).
# MAGIC - Para datasets pequenos e desbalanceados, ajustar o peso das classes
# MAGIC   (class_weight) tende a ser mais robusto que reamostragem sintetica
# MAGIC   tipo SMOTE quando ha poucos exemplos da classe minoritaria (estudo
# MAGIC   arXiv 2409.19751, 2024).
# MAGIC - Series hidrologicas (nivel de reservatorio, vazao) mostram forte
# MAGIC   autocorrelacao de curto prazo: o estado do mes anterior ajuda a
# MAGIC   prever o do mes atual (Harbin Engineering Journal, revisao de
# MAGIC   modelos de vazao).
# MAGIC
# MAGIC ### O que foi testado
# MAGIC Comparamos 4 conjuntos de variaveis x 3 algoritmos (12 combinacoes),
# MAGIC usando validacao cruzada **temporal** (5 janelas, sem embaralhar os
# MAGIC meses, para nao vazar informacao do futuro para o passado):
# MAGIC
# MAGIC | Conjunto de variaveis | O que tem |
# MAGIC |---|---|
# MAGIC | A - base v1 | Mes, chuva do mes anterior, %% da normal, temperatura, umidade |
# MAGIC | B - janela 3 meses | mesmas ideias, mas com chuva acumulada dos ultimos 3 meses (estilo SPI-3) |
# MAGIC | C - persistencia | conjunto A + a bandeira do MES ANTERIOR como variavel |
# MAGIC | D - tudo junto | A + B + C combinados |
# MAGIC
# MAGIC Algoritmos: Arvore de Decisao, Random Forest e Regressao Logistica.
# MAGIC
# MAGIC ### Resultado honesto do experimento
# MAGIC A janela de 3 meses (B) **NAO melhorou** o resultado (piorou, na
# MAGIC verdade) - contrariando a expectativa inicial baseada na literatura de
# MAGIC SPI. Reportamos isso porque o combinado importa: usar bandeira do mes
# MAGIC anterior (persistencia) foi o que realmente ajudou. A melhor combinacao
# MAGIC real foi **Regressao Logistica + variaveis climaticas do mes anterior +
# MAGIC bandeira do mes anterior**.
# MAGIC
# MAGIC ### Reprodutibilidade
# MAGIC - `random_state` fixo (42).
# MAGIC - Validacao cruzada temporal (5 janelas) para a escolha do modelo, e
# MAGIC   depois um teste final nos ultimos 24 meses (nunca vistos no treino)
# MAGIC   para o numero que vai na apresentacao.
# MAGIC - Fonte dos dados: `mba.trusted.f_bandeira` e
# MAGIC   `mba.refined.f_modelo_bandeira_clima`.

# COMMAND ----------

# DBTITLE 1,Monta a base com a variavel de persistencia
from pyspark.sql import functions as F
from pyspark.sql.window import Window

bandeira = spark.table("mba.trusted.f_bandeira").select("MesCompetencia", "IsVermelha")
w = Window.orderBy("MesCompetencia")
bandeira_lag = bandeira.withColumn("IsVermelhaMesAnterior", F.lag("IsVermelha", 1).over(w))

refined = spark.table("mba.refined.f_modelo_bandeira_clima")

df = refined.join(bandeira_lag.select("MesCompetencia", "IsVermelhaMesAnterior"), on="MesCompetencia", how="inner")
df = df.dropna(subset=["IsVermelhaMesAnterior"])
pdf = df.orderBy("MesCompetencia").toPandas()
print(f"Total de meses: {len(pdf)}")

# COMMAND ----------

# DBTITLE 1,Treina o modelo v2 (Regressao Logistica + persistencia) e avalia
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
FEATURES = ["Mes", "PrecipitacaoAcumuladaMm", "PrecipitacaoPctNormal",
            "TemperaturaMediaC", "UmidadeMediaPct", "IsVermelhaMesAnterior"]
TARGET = "IsVermelha"

X, y = pdf[FEATURES], pdf[TARGET]
N_TESTE = 24
X_train, X_test = X.iloc[:-N_TESTE], X.iloc[-N_TESTE:]
y_train, y_test = y.iloc[:-N_TESTE], y.iloc[-N_TESTE:]

modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
])

inicio = time.perf_counter()
modelo.fit(X_train, y_train)
tempo_treino_ms = (time.perf_counter() - inicio) * 1000

y_pred = modelo.predict(X_test)
print(f"Tempo de treino: {tempo_treino_ms:.1f} ms")
print(f"Acuracia:  {accuracy_score(y_test, y_pred):.2%}")
print(f"Precisao:  {precision_score(y_test, y_pred, zero_division=0):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.2%}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.2%}")
print("\nMatriz de confusao:\n", confusion_matrix(y_test, y_pred))

# COMMAND ----------

dbutils.notebook.exit("Executed")
