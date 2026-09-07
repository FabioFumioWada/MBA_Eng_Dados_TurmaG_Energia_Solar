# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## Treino do Modelo v3 - Previsao de Bandeira Vermelha (Sprint 4)
# MAGIC
# MAGIC ### Objetivo
# MAGIC Este notebook parte do modelo v2 (ver `02_regressao_logistica_persistencia_clima`)
# MAGIC e testa, de forma honesta e documentada, se o indice ONI da NOAA (que mede
# MAGIC a forca do El Nino/La Nina) melhora a previsao de bandeira vermelha, sem
# MAGIC inventar dados nem vazar informacao do futuro.
# MAGIC
# MAGIC ### Por que testar o El Nino/La Nina
# MAGIC O El Nino e o La Nina mudam o regime de chuvas no Brasil (tipicamente,
# MAGIC El Nino forte reduz chuva no Norte/Nordeste e reforca chuva no Sul), o
# MAGIC que pode afetar a geracao hidreletrica e, por consequencia, o acionamento
# MAGIC da bandeira tarifaria. A mesma logica de teste ja foi aplicada ao EAR
# MAGIC (nivel dos reservatorios) no experimento anterior.
# MAGIC
# MAGIC ### O que foi testado
# MAGIC Ampliamos o experimento de comparacao para 8 conjuntos de variaveis x 3
# MAGIC algoritmos (24 combinacoes), usando validacao cruzada **temporal** (5
# MAGIC janelas, sem embaralhar os meses):
# MAGIC
# MAGIC | Conjunto de variaveis | O que tem |
# MAGIC |---|---|
# MAGIC | A - base v1 | Mes, chuva do mes anterior, %% da normal, temperatura, umidade |
# MAGIC | B - janela 3 meses | mesmas ideias, com chuva acumulada dos ultimos 3 meses (estilo SPI-3) |
# MAGIC | C - persistencia | conjunto A + bandeira do MES ANTERIOR |
# MAGIC | D - tudo junto | A + B + C combinados |
# MAGIC | E - persistencia + EAR | C + nivel dos reservatorios (EAR) do mes anterior |
# MAGIC | F - so EAR + persistencia | Mes, IsVermelhaMesAnterior, EAR do mes anterior (sem clima) |
# MAGIC | G - persistencia + ONI | C + indice ONI (anomalia de temperatura do Pacifico) do mes anterior |
# MAGIC | H - so ONI + persistencia | Mes, IsVermelhaMesAnterior, ONI do mes anterior (sem clima) |
# MAGIC
# MAGIC ### Resultado honesto do experimento
# MAGIC O EAR (conjunto E) **NAO melhorou** o resultado da validacao cruzada em
# MAGIC relacao ao modelo v2 (persistencia + clima), mesmo tendo uma justificativa
# MAGIC teorica forte (reservatorio baixo deveria preceder bandeira vermelha).
# MAGIC Ja o ONI (conjunto G) **melhorou**, ainda que por uma margem pequena:
# MAGIC F1 medio de 0,478 contra 0,459 do v2, o melhor resultado entre as 24
# MAGIC combinacoes testadas. Por isso o modelo v3 usa persistencia + clima + ONI.
# MAGIC
# MAGIC Vale registrar uma observacao honesta: no teste final dos ultimos 24
# MAGIC meses (differente da validacao cruzada), o v3 chegou exatamente ao mesmo
# MAGIC resultado do v2 (mesma matriz de confusao). Ou seja, a vantagem do ONI
# MAGIC aparece na validacao cruzada (media de 5 janelas), mas nao mudou nenhuma
# MAGIC previsao especifica nesses ultimos 24 meses. Isso e reportado no
# MAGIC relatorio tecnico sem maquiagem.
# MAGIC
# MAGIC ### Reprodutibilidade
# MAGIC - `random_state` fixo (42).
# MAGIC - Validacao cruzada temporal (5 janelas) para a escolha do modelo, e
# MAGIC   depois um teste final nos ultimos 24 meses (nunca vistos no treino).
# MAGIC - Fonte dos dados: `mba.trusted.f_bandeira`, `mba.refined.f_modelo_bandeira_clima`
# MAGIC   e `mba.trusted.f_oni_enso` (indice ONI historico da NOAA).

# COMMAND ----------

# DBTITLE 1,Monta a base com persistencia da bandeira e indice ONI do mes anterior
from pyspark.sql import functions as F
from pyspark.sql.window import Window

bandeira = spark.table("mba.trusted.f_bandeira").select("MesCompetencia", "IsVermelha")
w = Window.orderBy("MesCompetencia")
bandeira_lag = bandeira.withColumn("IsVermelhaMesAnterior", F.lag("IsVermelha", 1).over(w))

oni = spark.table("mba.trusted.f_oni_enso").select(F.col("MesRef").alias("MesCompetencia"), "OniAnomC")
w_oni = Window.orderBy("MesCompetencia")
oni_lag = oni.withColumn("OniAnomMesAnterior", F.lag("OniAnomC", 1).over(w_oni))

refined = spark.table("mba.refined.f_modelo_bandeira_clima")

df = refined.join(bandeira_lag.select("MesCompetencia", "IsVermelhaMesAnterior"), on="MesCompetencia", how="inner")
df = df.join(oni_lag.select("MesCompetencia", "OniAnomMesAnterior"), on="MesCompetencia", how="inner")
df = df.dropna(subset=["IsVermelhaMesAnterior", "OniAnomMesAnterior"])
pdf = df.orderBy("MesCompetencia").toPandas()
print(f"Total de meses: {len(pdf)}")

# COMMAND ----------

# DBTITLE 1,Treina o modelo v3 (Regressao Logistica + persistencia + clima + ONI) e avalia
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
FEATURES = ["Mes", "PrecipitacaoAcumuladaMm", "PrecipitacaoPctNormal",
            "TemperaturaMediaC", "UmidadeMediaPct", "IsVermelhaMesAnterior",
            "OniAnomMesAnterior"]
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

# DBTITLE 1,Coeficientes do modelo (interpretabilidade)
import pandas as pd

coefs = modelo.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"Variavel": FEATURES, "Coeficiente": coefs}).sort_values("Coeficiente")
print(coef_df.to_string(index=False))

# COMMAND ----------

dbutils.notebook.exit("Executed")