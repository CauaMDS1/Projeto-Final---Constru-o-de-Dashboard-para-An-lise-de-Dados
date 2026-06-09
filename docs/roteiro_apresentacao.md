# Roteiro de apresentacao

## 1. Tema e objetivo

Tema: analise do mercado de jogos da Steam com base em catalogo de jogos e recomendacoes de usuarios.

Pergunta central:

> O que faz um jogo da Steam ser bem recomendado?

## 2. Fonte dos dados

- Kaggle: `games.csv`
- Kaggle: `recommendations.csv`
- Bonus: coleta automatica pela API publica do SteamSpy

## 3. Pipeline

1. Leitura dos CSVs com Pandas.
2. Correcao do cabecalho do `games.csv`.
3. Agregacao das 41 milhoes de recomendacoes por `app_id`.
4. Merge entre jogos e recomendacoes.
5. Tratamentos e transformacoes.
6. Geracao dos arquivos processados.
7. Construcao dos dashboards em Dash.

## 4. Tecnicas usadas

- Imputacao de valores.
- Normalizacao min-max.
- Padronizacao z-score.
- Codificacao one-hot.
- Discretizacao em faixas.
- Oversampling para balancear uma base supervisionada derivada.
- Remocao de features de baixa variancia.

## 5. Insights esperados

Sugestoes para comentar apos rodar o dashboard:

- comparar jogos gratuitos e pagos;
- verificar generos com maior taxa de recomendacao;
- observar jogos com muitas reviews e baixa satisfacao;
- comparar horas jogadas e recomendacao;
- analisar anos com maior crescimento de lancamentos.

## 6. Demonstracao

Mostrar:

1. Dashboard 1 como visao geral.
2. Dashboard 2 filtrando por genero e faixa de preco.
3. Tabela final de jogos em destaque.
4. Arquivo `pipeline_report.json` como evidencia do pipeline.

## 7. Bonus

Executar ou mostrar o script:

```powershell
python -m src.collect_steam_catalog
```

Explicacao:

> O projeto inclui uma coleta automatica via API publica do SteamSpy, salvando um novo arquivo bruto e integrando-o por `app_id`.
