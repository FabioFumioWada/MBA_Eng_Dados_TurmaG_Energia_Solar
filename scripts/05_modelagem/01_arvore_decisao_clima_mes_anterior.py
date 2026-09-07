# Databricks notebook source
# MAGIC %md
# MAGIC ## Treino do Modelo v1 - Previsao de Bandeira Vermelha (Sprint 4)
# MAGIC
# MAGIC ### Objetivo
# MAGIC Prever se a bandeira tarifaria de um mes sera **VERMELHA (1)** ou nao (0),
# MAGIC usando apenas o clima (chuva, temperatura, umidade) do **mes anterior**
# MAGIC das estacoes ligadas as usinas hidreletricas. E o modelo v1 combinado
# MAGIC (estrategia do Alberto): so hidreletrica + chuva + bandeira. EAR, ENA e
# MAGIC CMO ficam para uma proxima versao (v2), ja tratados no Databricks.
# MAGIC
# MAGIC ### Tecnica escolhida: Arvore de Decisao (Decision Tree Classifier)
# MAGIC Por que essa tecnica e nao algo mais complexo (rede neural, XGBoost)?
# MAGIC 1. **Poucos dados** (127 meses) - modelo complexo demais "decora" em vez
# MAGIC    de aprender (overfitting).
# MAGIC 2. **Interpretabilidade** - da pra desenhar a arvore e explicar as regras
# MAGIC    em portugues simples pra banca (ex.: "se choveu pouco e a umidade
# MAGIC    esta baixa, entao bandeira vermelha").
# MAGIC 3. Nao exige normalizar os dados.
# MAGIC
# MAGIC ### Reprodutibilidade
# MAGIC - `random_state` fixo (42) em tudo que tem aleatoriedade.
# MAGIC - Divisao **temporal** (nao aleatoria): treino = primeiros 103 meses,
# MAGIC   teste = ultimos 24 meses. Simula a situacao real (treinar com o
# MAGIC   passado, prever o futuro), sem vazar informacao do teste no treino.
# MAGIC - Fonte dos dados versionada: `mba.refined.f_modelo_bandeira_clima`
# MAGIC   (ver notebook `04_processamento_refined/05_f_modelo_bandeira_clima`).

# COMMAND ----------

# DBTITLE 1,Carrega a base do refined
df = spark.table("mba.refined.f_modelo_bandeira_clima").orderBy("MesCompetencia").toPandas()
print(f"Total de meses: {len(df)}")
df.head()

# COMMAND ----------

# DBTITLE 1,Escolhe as features e faz a divisao temporal treino/teste
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

RANDOM_STATE = 42
N_TESTE = 24  # ultimos 24 meses viram o "teste" (2 anos, com representatividade de vermelhas)

# QtdEstacoesUsadas fica de fora (quase nao varia, sem poder preditivo).
# PrecipitacaoMediaMm fica de fora (redundante com PrecipitacaoAcumuladaMm).
FEATURES = ["Mes", "PrecipitacaoAcumuladaMm", "PrecipitacaoPctNormal",
            "TemperaturaMediaC", "UmidadeMediaPct"]
TARGET = "IsVermelha"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test = X.iloc[:-N_TESTE], X.iloc[-N_TESTE:]
y_train, y_test = y.iloc[:-N_TESTE], y.iloc[-N_TESTE:]
meses_teste = df["MesCompetencia"].iloc[-N_TESTE:]

print(f"Treino: {len(X_train)} meses ({y_train.sum()} vermelhas)")
print(f"Teste:  {len(X_test)} meses ({y_test.sum()} vermelhas)")
print(f"Periodo de teste: {meses_teste.min()} a {meses_teste.max()}")

# COMMAND ----------

# DBTITLE 1,Treina o modelo
# class_weight="balanced": so ~22% dos meses sao vermelhos, isso evita que o
# modelo simplesmente "chute sempre nao-vermelha" pra parecer mais preciso.
modelo = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE)
modelo.fit(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Avalia no teste (meses que o modelo nunca viu)
y_pred = modelo.predict(X_test)

print(f"Acuracia: {accuracy_score(y_test, y_pred):.2%}")
print(f"Precisao: {precision_score(y_test, y_pred, zero_division=0):.2%}")
print(f"Recall:   {recall_score(y_test, y_pred, zero_division=0):.2%}")
print(f"F1-score: {f1_score(y_test, y_pred, zero_division=0):.2%}")
print("\nMatriz de confusao:\n", confusion_matrix(y_test, y_pred))
print("\nRegras aprendidas:\n", export_text(modelo, feature_names=FEATURES))

# COMMAND ----------

# DBTITLE 1,Desenha a arvore (visual para a apresentacao)
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(modelo, feature_names=FEATURES, class_names=["Nao-vermelha", "Vermelha"],
          filled=True, rounded=True, fontsize=10, ax=ax)
plt.title("Arvore de Decisao - Previsao de Bandeira Vermelha (v1)")
plt.tight_layout()
display(fig)

# COMMAND ----------

dbutils.notebook.exit("Executed")