# Contexto do Projeto

## O que e este projeto

Este repositorio implementa um projeto de **previsao de volatilidade realizada do Bitcoin** a partir de dados de preco em frequencia de **5 minutos**.  
O foco e comparar abordagens de duas familias:

- **Modelo econometrico HAR** (baseline interpretable).
- **Modelo Transformer PatchTST** (deep learning para series temporais).

## Objetivo principal

Construir um pipeline profissional, modular e reproduzivel para:

1. Preparar os dados de mercado.
2. Treinar e avaliar HAR e PatchTST.
3. Comparar desempenho com metricas consistentes.
4. Evoluir o modelo com dados adicionais (ex.: sentimento) e novos baselines.

## Estado atual do projeto

- **Etapa 1 (modularizacao) concluida**: a logica saiu dos notebooks e foi organizada em modulos Python + CLI + testes.
- O pipeline atual ja permite preparar dados, treinar HAR, treinar PatchTST e comparar resultados.
- As proximas evolucoes planejadas sao:
  - **Etapa 2 (prioritaria):** integrar dados de sentimento ao PatchTST.
  - **Etapa 3:** testar outros modelos (Transformers e/ou ML classico/recorrente).

## Estrutura do repositorio

```text
data/
  raw/                   # dados brutos (ex.: BTCUSDT_5m.txt)
  interim/               # artefatos intermediarios de preparo/splits
  processed/             # previsoes e metricas finais

src/
  config.py              # caminhos, colunas, constantes e hiperparametros
  dataset.py             # ingestao, features, split temporal e datasets
  metrics.py             # metricas de avaliacao e persistencia
  pipeline.py            # orquestracao ponta a ponta
  cli.py                 # interface de linha de comando (Typer)
  models/
    har.py               # treinamento/avaliacao do baseline HAR
    patchtst.py          # treinamento/avaliacao do PatchTST
    transformer.py       # utilitarios legados de avaliacao visual

tests/
  test_dataset.py
  test_metrics.py
```

## Como o projeto funciona (fluxo)

1. **Ingestao e preparo** dos dados de preco BTC 5m.
2. **Feature engineering** de volatilidade realizada e variaveis derivadas.
3. **Split temporal cronologico** (treino/validacao/teste) com preservacao de contexto.
4. **Treino e avaliacao** dos modelos (HAR e PatchTST).
5. **Persistencia de resultados** em `data/processed/` para comparacao.

## Como executar

```bash
pip install -r requirements.txt
python -m src.cli prepare
python -m src.cli train-har
python -m src.cli train-patchtst
python -m src.cli compare
```

## Papel dos notebooks

Os notebooks continuam como apoio para exploracao e analise, mas a logica central do projeto deve permanecer em `src/` para manter:

- modularidade,
- testabilidade,
- clareza arquitetural,
- reproducibilidade.

## Convencoes importantes

- Nomeacao e dominio em portugues (ex.: `RepositorioDados`, `executar`).
- Parametros e caminhos centralizados em `src/config.py`.
- Layout de artefatos:
  - `data/raw` para entrada,
  - `data/interim` para intermediarios,
  - `data/processed` para saida final,
  - `models/` para artefatos de modelos.

## Resumo de uso em novas sessoes

Em qualquer nova sessao, use este documento como ponto de entrada para entender rapidamente:

- o problema de negocio/tecnico,
- a arquitetura atual,
- o fluxo operacional,
- e as proximas evolucoes esperadas.
