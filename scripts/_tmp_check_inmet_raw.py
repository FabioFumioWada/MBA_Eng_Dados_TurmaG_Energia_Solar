# Databricks notebook source
resultado = spark.sql("""
    SELECT ano, COUNT(*) AS registros, COUNT(DISTINCT codigo_wmo) AS estacoes
    FROM mba.raw.clima_inmet
    GROUP BY ano
    ORDER BY ano
""").collect()
for r in resultado:
    print(r)
total = spark.sql("SELECT COUNT(*) AS c FROM mba.raw.clima_inmet").collect()[0]['c']
print("TOTAL:", total)
dbutils.notebook.exit(str(total))
