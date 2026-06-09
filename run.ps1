[CmdletBinding()]
param(
    [switch]$ForcePrepare,
    [switch]$SkipPrepare,
    [switch]$CollectBonus,
    [switch]$NoBrowser,
    [int]$FastSampleRows = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SetupScript = Join-Path $ProjectRoot "setup.ps1"
$GamesCsv = Join-Path $ProjectRoot "data\raw\games.csv"
$RecommendationsCsv = Join-Path $ProjectRoot "data\raw\recommendations.csv"
$AnalyticsCsv = Join-Path $ProjectRoot "data\processed\steam_analytics.csv"

function Require-RawData {
    $Missing = @()
    if (-not (Test-Path $GamesCsv)) { $Missing += "data\raw\games.csv" }
    if (-not (Test-Path $RecommendationsCsv)) { $Missing += "data\raw\recommendations.csv" }

    if ($Missing.Count -gt 0) {
        Write-Host ""
        Write-Host "Faltam arquivos de dados brutos." -ForegroundColor Yellow
        Write-Host "Coloque estes arquivos dentro da pasta do projeto:" -ForegroundColor Yellow
        foreach ($Item in $Missing) { Write-Host "  - $Item" }
        Write-Host ""
        Write-Host "Depois rode novamente: .\run.ps1"
        exit 1
    }
}

if (-not (Test-Path $VenvPython)) {
    & $SetupScript
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Require-RawData

if ($CollectBonus) {
    Write-Host "Coletando dados extras da API SteamSpy..." -ForegroundColor Cyan
    & $VenvPython -m src.collect_steam_catalog
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($SkipPrepare -and -not (Test-Path $AnalyticsCsv)) {
    Write-Host "Nao existem dados processados em data\processed." -ForegroundColor Yellow
    Write-Host "Remova -SkipPrepare ou rode .\run_prepare.ps1 antes de abrir o dashboard."
    exit 1
}

$NeedPrepare = $ForcePrepare -or (-not (Test-Path $AnalyticsCsv))
if ($NeedPrepare -and -not $SkipPrepare) {
    Write-Host "Preparando dados. Isso pode demorar no primeiro uso..." -ForegroundColor Cyan
    $PrepareArgs = @("-m", "src.prepare_data")
    if ($FastSampleRows -gt 0) {
        $PrepareArgs += @("--max-recommendation-rows", "$FastSampleRows")
    }
    & $VenvPython @PrepareArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "Dados processados encontrados. Pulando preparacao." -ForegroundColor Green
}

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 3
        Start-Process "http://127.0.0.1:8050"
    } | Out-Null
}

Write-Host "Abrindo dashboard em http://127.0.0.1:8050" -ForegroundColor Green
Write-Host "Use Ctrl+C para encerrar o servidor."
& $VenvPython -m src.app
