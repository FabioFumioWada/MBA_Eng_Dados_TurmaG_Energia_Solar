# Bandeira Tarifária, Hidrologia e Clima - MBA Engenharia de Dados

Projeto acadêmico do MBA em Engenharia de Dados (Mackenzie, Turma G) que investiga a relação entre a bandeira tarifária de energia elétrica no Brasil (ANEEL) e os indicadores hidrológicos, climáticos e operacionais que ajudam a explicar o seu comportamento: nível dos reservatórios (EAR), vazão dos rios (ENA), custo marginal de operação (CMO) e carga de energia, todos do ONS, além do clima diário (INMET), com apoio do cadastro de usinas hidrelétricas (ANEEL/SIGA) como base geográfica.

> Este repositório substitui o escopo original do grupo, que era bandeira tarifária e geração distribuída solar (disponível no repositório anterior, [MACK_MBA_Eng_Dados_TurmaG_Energia_Solar](https://github.com/FabioFumioWada/MACK_MBA_Eng_Dados_TurmaG_Energia_Solar)). O tema foi redefinido para hidrologia e clima, mantendo a bandeira tarifária como eixo central da análise.

## Integrantes do grupo

- Alberto Oliveira Chaves, RA 1015803
- Fabio Fumio Wada, RA 10741479
- Laiane Ressurreição, RA 10739799
- Sweeli Suzuki, RA 10423319
- Tatiane Silva Santos, RA 10747108

## Sumário

- [Visão geral do pipeline](#visão-geral-do-pipeline)
- [Fontes de dados](#fontes-de-dados)
- [Arquitetura no Databricks](#arquitetura-no-databricks)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como reproduzir](#como-reproduzir)
- [Principais achados até agora](#principais-achados-até-agora)
- [Limitações e próximos passos](#limitações-e-próximos-passos)
- [Documentação complementar](#documentação-complementar)

## Visão geral do pipeline

Todo dado passa por três camadas, dentro do catálogo `mba` do Unity Catalog (Databricks Free/Community, cluster serverless):

| Camada | O que é | Formato |
|---|---|---|
| `stage` | Arquivo bruto exatamente como veio da fonte (CSV, ZIP), sem qualquer transformação | Arquivo original, em Volume |
| `raw` | Mesmo conteúdo do stage, apenas convertido para um formato de tabela | `.parquet` / Delta Table |
| `trusted` | Dado tipado, padronizado e deduplicado, com carga incremental idempotente (`MERGE INTO`) | Delta Table |

Esse padrão de três camadas foi definido a partir do notebook de referência do grupo para o cadastro de usinas hidrelétricas e foi replicado da mesma forma para as sete novas fontes de dados adicionadas neste projeto.

## Fontes de dados

| Fonte | Fornecedor | Cobertura | Linhas (raw/trusted) | Coleta automatizada no Databricks? |
|---|---|---|---|---|
| Cadastro de usinas hidrelétricas | ANEEL/SIGA | Snapshot atual | referência de padrão | Sim |
| Bandeira tarifária (acionamento mensal) | ANEEL | jan/2015 a ago/2026 | 140 | Não, o portal bloqueia o Databricks; upload manual do CSV |
| EAR, nível dos reservatórios | ONS | 2000 a 2026 | 38.976 | Sim |
| ENA, vazão dos rios | ONS | 2000 a 2026 | 38.976 | Sim |
| CMO, custo marginal de operação | ONS | 2005 a 2026 | 4.524 | Sim |
| Carga de energia | ONS | 2000 a 2026 | 38.973 | Sim |
| Clima diário | INMET | 2016 a 2026 | 53.275.560 | Não, o portal bloqueia o Databricks; upload manual do ZIP |
| PLD, preço de curto prazo | CCEE | apenas 2026 disponível | pendente | Não, o portal da CCEE retorna erro 403 (bloqueado); dataset em espera |

Descoberta técnica desta etapa: o domínio `ons-aws-prod-opendata.s3.amazonaws.com` (ONS) é acessível diretamente pelo cluster serverless do Databricks, o que permitiu automatizar toda a coleta de EAR, ENA, CMO e carga. Já `dadosabertos.aneel.gov.br` (bandeira) e `portal.inmet.gov.br` (clima) bloqueiam a requisição vinda do Databricks. Para essas duas fontes, o notebook de stage documenta o processo e o arquivo precisa ser baixado localmente, fora do Databricks, e depois enviado ao Volume antes de rodar raw e trusted.

Nesta etapa também ampliamos a profundidade histórica do clima INMET, que antes cobria apenas 2024 a 2026 e agora cobre 2016 a 2026, para dar mais anos de histórico ao futuro modelo preditivo do risco de bandeira.

## Arquitetura no Databricks

- Workspace: `dbc-7274ec51-331f.cloud.databricks.com`
- Catálogo (Unity Catalog): `mba`, com os esquemas `stage`, `raw` e `trusted`
- Processamento: notebooks PySpark e Spark SQL, executados via API REST (`workspace/import` seguido de `jobs/runs/submit`)
- Idempotência: a camada trusted usa `MERGE INTO` com chave substituta `IDENTITY`, então reexecutar qualquer notebook não duplica linhas
- Versionamento: Git e GitHub, sincronizados manualmente via Databricks Repos (aba Git). CI/CD com GitHub Actions ainda não foi implementado, é o próximo passo de automação do pipeline

## Estrutura do repositório

```
MBA_Eng_Dados_TurmaG_Energia_Solar/
├── scripts/
│   ├── 00_setup_estrutura_energia.ipynb        -> cria a estrutura de catálogo e esquemas no Unity Catalog
│   ├── 01_ingestao_stage/                      -> um notebook por fonte, camada stage
│   │   ├── 01_cadastro_usinas.ipynb
│   │   ├── 02_geracao_energia.ipynb
│   │   ├── 03_inmet.py
│   │   ├── 04_ear.py
│   │   ├── 05_bandeira.py
│   │   ├── 06_ena.py
│   │   ├── 07_cmo.py
│   │   └── 08_carga.py
│   ├── 02_processamento_raw/                   -> mesmas 8 fontes, convertendo stage em parquet
│   └── 03_processamento_trusted/               -> mesmas 8 fontes, convertendo raw em Delta Table (MERGE INTO)
├── src/
│   └── ingest_dados_abertos.py                 -> script de apoio para ingestão de dados abertos
└── README.md                                   -> este arquivo
```

Essa organização segue a mesma lógica de camadas (stage, raw, trusted) pedida para o notebook de referência de usinas hidrelétricas, replicada para todas as fontes novas do projeto. Cada uma das três pastas de scripts tem um notebook por fonte, sempre com o mesmo padrão de nome, o que facilita achar rapidamente qual notebook processa qual dado em qual camada.

## Como reproduzir

1. Clonar o repositório dentro do Databricks Repos (aba Git), na pasta `/Users/<seu usuário>/MBA_Eng_Dados_TurmaG_Energia_Solar/`.
2. Rodar `scripts/00_setup_estrutura_energia.ipynb` uma única vez, para garantir que o catálogo `mba` e os esquemas `stage`, `raw` e `trusted` existam.
3. Para cada fonte, rodar os três notebooks em ordem: `01_ingestao_stage/<fonte>`, depois `02_processamento_raw/<fonte>` e por fim `03_processamento_trusted/<fonte>`.
4. Para bandeira tarifária e clima INMET, baixar o arquivo manualmente (fora do Databricks) e enviá-lo ao Volume de stage indicado no próprio notebook, antes do passo 3 acima.
5. Validar comparando a contagem de linhas da tabela trusted com o CSV local consolidado da mesma fonte, que usa a mesma lógica de limpeza em Python e pandas, como referência cruzada.

## Principais achados até agora

Em um painel mensal consolidado de 140 meses (jan/2015 a ago/2026), cruzando o nível ordinal da bandeira (Verde=0, Amarela=1, Vermelha P1=2, Vermelha P2=3, Escassez Hídrica=4) com os indicadores nacionais:

| Indicador | Correlação de Pearson com o nível da bandeira |
|---|---|
| Valor adicional da bandeira (R$/MWh) | +0,93 |
| CMO médio nacional | +0,45 |
| EAR nacional (%) | -0,33 |
| Carga total nacional | -0,14 |
| ENA nacional (% da MLT) | -0,05 |

- Reservatório mais baixo (EAR) tende a coincidir com bandeira mais alta, mas a força da correlação é apenas moderada.
- O custo de operação (CMO) tem correlação mais forte com a bandeira do que o próprio nível do reservatório.
- A vazão dos rios (ENA) tem correlação quase nula em nível nacional agregado, um provável sinal de que o efeito aparece por subsistema e com defasagem temporal, e não na média nacional simultânea.
- Distribuição histórica completa da bandeira: Verde 47,9%, Vermelha P1 18,6%, Amarela 17,1%, Vermelha P2 10,7% e Escassez Hídrica 5,7%.

## Limitações e próximos passos

- CI/CD (GitHub Actions e testes automatizados) ainda não implementado. A sincronização com o GitHub é manual, via Databricks Repos.
- PLD (CCEE) só tem 2026 disponível, o portal está bloqueado (erro 403) para os anos anteriores.
- As correlações desta etapa são nacionais e simultâneas. A próxima etapa vai repetir a análise por subsistema elétrico e com defasagem de 1 a 3 meses.
- Próximos passos planejados: modelo preditivo do nível de bandeira, automação do pipeline (agendamento e CI/CD) e dashboard de acompanhamento.

## Documentação complementar

Cada fonte de dados tem um README técnico próprio, com a metodologia detalhada de limpeza e a descrição do notebook Databricks correspondente:

- README, Processamento Clima INMET
- README, Notebooks INMET no Databricks (stage, raw, trusted)
- README, Processamento EAR Reservatórios
- README, Processamento ENA Vazão dos Rios
- README, Processamento CMO Custo de Operação
- README, Processamento Carga de Energia
- README, Processamento PLD CCEE

## Fontes oficiais

- ANEEL, Bandeiras Tarifárias: https://dadosabertos.aneel.gov.br/dataset/bandeiras-tarifarias
- ANEEL/SIGA, Sistema de Informações de Geração: https://dados.aneel.gov.br/dataset/siga
- ONS, EAR Diário por Subsistema: https://dados.ons.org.br/dataset/ear-diario-por-subsistema
- ONS, ENA Diário por Subsistema: https://dados.ons.org.br/dataset/ena-diario-por-subsistema
- ONS, Custo Marginal de Operação (CMO) Semanal: https://dados.ons.org.br/dataset/cmo-semanal
- ONS, Carga de Energia: https://dados.ons.org.br/dataset/carga-energia
- INMET, Dados Históricos: https://portal.inmet.gov.br/dadoshistoricos
- CCEE, PLD Média Diária: https://dadosabertos.ccee.org.br/dataset/pld_media_diaria
- Repositório GitHub do grupo (atual): https://github.com/FabioFumioWada/MBA_Eng_Dados_TurmaG_Energia_Solar
- Repositório GitHub do grupo (versão anterior, tema de GD Solar): https://github.com/FabioFumioWada/MACK_MBA_Eng_Dados_TurmaG_Energia_Solar

---
MBA em Engenharia de Dados, Mackenzie, Turma G (2026).
