# Steam Insights - Projeto de Ciencia de Dados

Projeto em Python para analisar jogos da Steam e recomendacoes de usuarios. Ele foi montado para atender ao trabalho final com:

- dois arquivos principais integrados: `games.csv` e `recommendations.csv`;
- pipeline com aquisicao, integracao, limpeza, transformacao e EDA;
- dois dashboards em Dash;
- coleta automatica de dados para o bonus;
- tecnicas vistas em aula: imputacao, normalizacao, padronizacao, codificacao, discretizacao, oversampling, PCA e selecao de baixa variancia.

## Tema

**O que faz um jogo da Steam ser bem recomendado?**

O projeto cruza caracteristicas dos jogos, como genero, preco, plataforma, data de lancamento e popularidade, com comportamento de usuarios nas reviews, como recomendacao positiva/negativa, horas jogadas, votos de utilidade e votos de humor.

## Estrutura

```text
steam_recommendations_dashboard/
  data/
    raw/
    processed/
  docs/
  src/
    app.py
    collect_steam_catalog.py
    prepare_data.py
    utils.py
  requirements.txt
  README.md
```

## Dados usados

Os dados brutos usados no projeto estao em `data/raw`:

```text
data/raw/games.csv
data/raw/recommendations.csv
data/raw/steam_catalog_api.csv
```

O pipeline usa esses arquivos locais por padrao. Caso queira rodar com arquivos em outro local, defina as variaveis de ambiente:

```powershell
$env:GAMES_CSV="C:\caminho\games.csv"
$env:RECOMMENDATIONS_CSV="C:\caminho\recommendations.csv"
```

## Como instalar e executar

### Pre-requisitos

Antes de rodar o projeto, tenha instalado:

- Python 3.11 ou superior;
- Git, caso voce va clonar pelo terminal;
- PowerShell, no Windows.

Depois de clonar o repositorio, entre na pasta do projeto:

```powershell
cd steam_recommendations_dashboard
```

### 1. Colocar os dados brutos

Os arquivos grandes nao ficam no GitHub por causa do tamanho. Depois do clone, coloque os CSVs nesta pasta:

```text
data/raw/games.csv
data/raw/recommendations.csv
```

A estrutura deve ficar assim:

```text
steam_recommendations_dashboard/
  data/
    raw/
      games.csv
      recommendations.csv
```

O arquivo `steam_catalog_api.csv` e opcional, porque pode ser gerado pelo script do bonus.

### 2. Jeito mais facil no Windows

Rode apenas:

```powershell
.\run.ps1
```

Esse comando faz automaticamente:

- cria o ambiente virtual `.venv`, se ele ainda nao existir;
- instala as dependencias do `requirements.txt`;
- verifica se `games.csv` e `recommendations.csv` estao em `data/raw`;
- prepara os dados tratados em `data/processed`, se ainda nao existirem;
- abre o navegador em `http://127.0.0.1:8050`;
- inicia o dashboard em Dash.

Se o PowerShell bloquear a execucao de scripts, rode:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Tambem e possivel dar duplo clique em:

```text
run.bat
```

### 3. Acessar o dashboard

Quando o servidor iniciar, abra no navegador:

```text
http://127.0.0.1:8050
```

Para parar o servidor, volte ao terminal e pressione:

```text
Ctrl+C
```

### 4. Comandos uteis

Preparar tudo do zero novamente:

```powershell
.\run.ps1 -ForcePrepare
```

Rodar com uma amostra menor para teste rapido:

```powershell
.\run.ps1 -FastSampleRows 500000
```

Rodar a coleta automatica do bonus antes de abrir o dashboard:

```powershell
.\run.ps1 -CollectBonus
```

Abrir somente o dashboard, usando os dados ja processados:

```powershell
.\run_dashboard.ps1
```

Executar somente o pipeline de preparacao:

```powershell
.\run_prepare.ps1
```

Executar somente a coleta do bonus:

```powershell
.\run_collect_bonus.ps1
```

### 5. Execucao manual

Se preferir executar sem os scripts auxiliares:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.prepare_data
python -m src.app
```

Depois abra:

```text
http://127.0.0.1:8050
```

### 6. Problemas comuns

Se aparecer erro dizendo que os dados brutos nao foram encontrados, confira se os arquivos estao exatamente nestes caminhos:

```text
data/raw/games.csv
data/raw/recommendations.csv
```

Se as dependencias nao instalarem, atualize o `pip` e tente novamente:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se quiser refazer os arquivos processados, apague os CSVs de `data/processed` ou rode:

```powershell
.\run.ps1 -ForcePrepare
`$([Environment]::NewLine)
## Dashboards

### Dashboard 1 - Visao Geral

Painel executivo com:

- total de jogos;
- total de reviews integradas;
- taxa global de recomendacao;
- cobertura da integracao;
- preco medio;
- lancamentos por ano;
- generos mais comuns;
- recomendacao media por faixa de preco;
- disponibilidade por plataforma.

### Dashboard 2 - Exploracao Interativa

Painel de exploracao com filtros:

- genero;
- plataforma;
- faixa de preco;
- ano de lancamento;
- minimo de reviews.

Visualizacoes:

- dispersao de preco vs taxa de recomendacao;
- jogos com mais reviews;
- boxplot de recomendacao por faixa de preco;
- linha de recomendacao media por ano;
- histograma de horas jogadas;
- recomendacao media por genero;
- grafico PCA `PC1 x PC2`;
- tabela com jogos em destaque.

## Tecnicas aplicadas

### Imputacao

- Campos textuais ausentes recebem `Unknown`.
- Preco ausente recebe a mediana.
- Metacritic igual a zero ou ausente recebe a mediana dos valores validos em uma coluna separada.

### Normalizacao

Colunas `norm_log_*` usam `log1p` seguido de min-max scaling.

### Padronizacao

Colunas `std_log_*` usam `log1p` seguido de z-score.

### Codificacao

O arquivo `model_prepared_oversampled.csv` contem:

- generos mais frequentes codificados como variaveis binarias;
- faixas de preco com one-hot encoding;
- plataformas como 0/1.

### Discretizacao

Foram criadas faixas:

- `price_band`;
- `review_volume_band`;
- `hours_band`;
- `recommendation_rate_band`.

### Oversampling

Foi criada uma base supervisionada com o alvo `target_high_recommendation`. A classe minoritaria e balanceada por oversampling aleatorio com reposicao.

### PCA

O PCA foi aplicado com NumPy usando SVD sobre as features normalizadas, padronizadas e codificadas. A tecnica gera:

- `pca_components.csv`, com `PC1`, `PC2` e `PC3` por jogo;
- `pca_explained_variance.csv`, com variancia explicada por componente.

O segundo dashboard inclui um grafico `PC1 x PC2` para visualizar agrupamentos de jogos por perfil.

### Selecao de features

Antes do oversampling, variaveis codificadas com variancia muito baixa sao removidas.

## Bonus

O script `src.collect_steam_catalog.py` faz uma coleta automatica na API publica do SteamSpy e gera:

```text
data/raw/steam_catalog_api.csv
```

Esse arquivo e integrado ao pipeline por `app_id`, comprovando coleta automatica e integracao adicional.

## Arquivos gerados

```text
data/processed/games_clean.csv
data/processed/recommendations_aggregated.csv
data/processed/review_trends_by_year.csv
data/processed/steam_analytics.csv
data/processed/model_prepared_oversampled.csv
data/processed/pca_components.csv
data/processed/pca_explained_variance.csv
data/processed/pipeline_report.json
```



