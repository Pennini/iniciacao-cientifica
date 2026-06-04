# Planejamento do Projeto

## Finalidade deste documento

Este arquivo define o plano completo do projeto para garantir continuidade entre sessoes.  
Sempre que uma nova sessao iniciar, este deve ser o documento de referencia para entender:

- o escopo tecnico,
- o estado atual,
- a ordem de execucao das etapas,
- os criterios de sucesso,
- e como operar o pipeline.

## Resumo executivo

Projeto de previsao de volatilidade realizada do Bitcoin (dados de preco em 5 minutos), comparando:

1. Baseline econometrico **HAR**.
2. Modelo Transformer **PatchTST**.

O projeto foi estruturado para reproducibilidade, modularidade e comparacao justa entre modelos.

## Estado atual (checkpoint de referencia)

- **Etapa 1 concluida**: modularizacao da logica em `src/`, comandos CLI e testes iniciais.
- O fluxo atual de dados, treino e avaliacao ja funciona para HAR e PatchTST.
- O foco do projeto agora e:
  - **Etapa 2 (mais importante):** integrar dados de sentimento no PatchTST.
  - **Etapa 3:** ampliar benchmark com outros modelos.

## Objetivo geral do projeto

Construir um framework de forecast de volatilidade com:

- pipeline consistente de dados,
- protocolos de avaliacao comparaveis,
- capacidade de incorporar novas fontes (sentimento),
- e evolucao para benchmark multi-modelo.

## Arquitetura operacional

```text
data/
  raw/                 # entrada bruta (BTCUSDT_5m.txt e futuras fontes)
  interim/             # dados preparados e splits
  processed/           # metricas, previsoes e comparativos finais

src/
  config.py            # contratos centrais (paths, colunas, parametros)
  dataset.py           # ingestao, features, split temporal e datasets
  metrics.py           # metricas e persistencia de resultados
  pipeline.py          # orquestracao fim a fim
  cli.py               # comandos operacionais
  models/
    har.py
    patchtst.py
    transformer.py     # utilitarios legados

tests/
  test_dataset.py
  test_metrics.py
```

## Protocolo padrao de execucao

```bash
pip install -r requirements.txt
python -m src.cli prepare
python -m src.cli train-har
python -m src.cli train-patchtst
python -m src.cli compare
```

## Roadmap oficial (ordem cronologica obrigatoria)

## Etapa 1 - Organizar e profissionalizar o repositorio (CONCLUIDA)

### Objetivo
Transformar logica espalhada em notebooks em pipeline modular, testavel e executavel por CLI.

### Resultado esperado
- Codigo central em `.py`.
- Separacao clara de responsabilidades por modulo.
- Testes iniciais para partes criticas.
- Fluxo reproduzivel sem depender de notebook para execucao principal.

### Status
**Concluida.**

---

## Etapa 2 - Integrar dados de sentimento ao PatchTST (PRIORIDADE MAXIMA)

### Objetivo
Incorporar `DadosSentimentoBTC.xlsx` como fonte adicional de sinais para treino e avaliacao do PatchTST.

### Escopo funcional
1. Ler e padronizar o dataset de sentimento.
2. Mapear colunas candidatas de sentimento (segundo documentacao LSEG).
3. Alinhar temporalmente com a base de volatilidade sem vazamento.
4. Tratar lacunas de dados de forma explicita (imputacao conservadora + indicadores de ausencia quando necessario).
5. Treinar novamente o PatchTST com novas features.
6. Comparar:
   - PatchTST baseline (sem sentimento),
   - PatchTST + sentimento.

### Entregaveis tecnicos
- Extensoes no pipeline de `src/dataset.py` para ingestao/merge temporal de sentimento.
- Parametros e flags em `src/config.py` para controlar features de sentimento.
- Ajustes no treino em `src/models/patchtst.py` e orquestracao em `src/pipeline.py`.
- Saidas comparativas em `data/processed/` com metricas e previsoes.

### Criterios de sucesso
- Pipeline reproduzivel do inicio ao fim.
- Comparacao justa (mesmo split temporal e metricas).
- Decisao tecnica clara com base nos resultados: ganho, empate ou perda.

### Riscos tecnicos principais
- Frequencia temporal diferente entre dados de preco e sentimento.
- Colunas de sentimento com alta esparsidade.
- Vazamento temporal no merge/feature engineering.

### Mitigacao esperada
- Regras de merge explicitamente causais.
- Validacoes de cobertura por periodo (train/validation/test).
- Registro claro de features usadas em cada experimento.

---

## Etapa 3 - Benchmark expandido de modelos

### Objetivo
Avaliar alternativas ao PatchTST para aumentar robustez de conclusoes.

### Candidatos recomendados
1. **Chronos** (prioridade entre transformers adicionais).
2. **LSTM** e **GRU** (baselines recorrentes).
3. **XGBoost** (baseline tabular forte para series com features).

### Escopo funcional
- Padronizar protocolo de treino/avaliacao para todos os modelos.
- Garantir comparacao no mesmo horizonte, split e metricas.
- Consolidar ranking final de desempenho.

### Entregaveis tecnicos
- Modulos/model wrappers para cada familia.
- Saida comparativa unica com metricas agregadas.
- Registro de trade-offs: acuracia, custo computacional e interpretabilidade.

### Criterios de sucesso
- Benchmark reproduzivel.
- Conclusao objetiva sobre melhor candidato para proxima fase do projeto.

## Metricas de avaliacao padrao

As comparacoes devem manter o mesmo conjunto de metricas do projeto:

- RMSE
- MAE
- MAPE
- sMAPE

Sempre comparar usando exatamente o mesmo recorte temporal para evitar viés.

## Regras de continuidade entre sessoes

1. Ler primeiro:
   - `CONTEXTO_PROJETO.md`
   - `PLANEJAMENTO_PROJETO.md`
2. Confirmar em qual etapa o projeto esta.
3. Executar apenas a etapa vigente (na ordem 1 -> 2 -> 3).
4. Ao finalizar uma etapa, atualizar este arquivo com:
   - status,
   - entregaveis concluidos,
   - pendencias.

## Regra de decisao do projeto

A tomada de decisao deve ser orientada por evidencia:

- Sem ganho consistente com sentimento -> manter baseline e revisar features.
- Com ganho consistente -> promover pipeline com sentimento como novo padrao.
- Em benchmark expandido -> priorizar modelo com melhor equilibrio entre desempenho e custo operacional.

## Definicao de pronto por etapa

Uma etapa so deve ser marcada como concluida quando houver:

1. Codigo modular implementado.
2. Execucao CLI reproduzivel.
3. Resultados persistidos em `data/processed/`.
4. Comparacao clara com metricas.
5. Documentacao atualizada neste arquivo.

## Referencias internas

- Contexto geral: `CONTEXTO_PROJETO.md`
- Visao de uso rapido: `README.md`

