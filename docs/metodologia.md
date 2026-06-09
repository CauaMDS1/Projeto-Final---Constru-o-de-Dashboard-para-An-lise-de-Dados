# Metodologia

Este documento resume como o projeto aplica as etapas pedidas no trabalho e as tecnicas vistas nos materiais de apoio do professor.

## 1. Aquisicao de dados

Foram usados dois arquivos publicos do Kaggle:

- `games.csv`: catalogo dos jogos, precos, plataformas, generos, tags e metricas agregadas.
- `recommendations.csv`: recomendacoes/reviews de usuarios, horas jogadas e data da review.

Para o bonus, o script `src.collect_steam_catalog.py` coleta automaticamente um catalogo atualizado pela API publica do SteamSpy.

## 2. Integracao de dados

A integracao principal usa:

```python
analytics = games.merge(recommendations_aggregated, on="app_id", how="left")
```

O `recommendations.csv` e agregado por `app_id` antes do merge, evitando que o dashboard precise carregar mais de 41 milhoes de linhas.

## 3. Limpeza e tratamento

Tratamentos aplicados:

- correcao do cabecalho `DiscountDLC count`, separado em `Discount` e `DLC count`;
- conversao de datas;
- conversao de colunas numericas;
- remocao de duplicidades por `app_id`;
- tratamento de campos textuais ausentes;
- criacao de flags de ausencia.

## 4. Transformacao

Variaveis criadas:

- `owners_low`, `owners_high`, `owners_mid`;
- `total_steam_reviews`;
- `steam_positive_rate`;
- `external_recommendation_rate`;
- `avg_review_hours`;
- `platform_count`;
- `primary_genre`;
- `popularity_score`.

## 5. Tecnicas de tratamento e preparacao

### Imputacao

A imputacao e usada para preencher dados ausentes ou codificados como zero quando zero indica ausencia de informacao.

### Normalizacao e padronizacao

Variaveis muito assimetricas passam por `log1p` antes de:

- min-max scaling;
- z-score.

### Codificacao

Foram criadas variaveis binarias para generos, plataformas e faixas de preco.

### Discretizacao

As variaveis continuas foram transformadas em faixas interpretaveis, facilitando graficos comparativos.

### Oversampling

O oversampling foi aplicado em uma base supervisionada derivada, com a classe `target_high_recommendation`.

### Baixa variancia

Features codificadas com variancia quase nula foram removidas antes de gerar a base balanceada.

### PCA

O PCA foi aplicado com NumPy/SVD sobre as features normalizadas, padronizadas e codificadas. Foram gerados os componentes `PC1`, `PC2` e `PC3`, alem da tabela de variancia explicada por componente.

## 6. Analise exploratoria

Os dashboards foram organizados para responder perguntas como:

- generos mais comuns tambem sao os mais bem recomendados?
- jogos gratuitos recebem recomendacoes diferentes dos jogos pagos?
- jogos com mais horas jogadas tendem a ser mais recomendados?
- a taxa de recomendacao varia por ano de lancamento?
- quais jogos concentram mais reviews?

## 7. Comunicacao visual

O projeto separa uma visao executiva e uma visao exploratoria:

- Dashboard 1: resumo rapido.
- Dashboard 2: filtros e comparacoes detalhadas.
