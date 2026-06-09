[CmdletBinding()]
param(
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$AnalyticsCsv = Join-Path $ProjectRoot "data\processed\steam_analytics.csv"

if (-not (Test-Path $VenvPython)) {
    & (Join-Path $ProjectRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path $AnalyticsCsv)) {
    Write-Host "Dados processados nao encontrados. Preparando agora..." -ForegroundColor Cyan
    & (Join-Path $ProjectRoot "run_prepare.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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
