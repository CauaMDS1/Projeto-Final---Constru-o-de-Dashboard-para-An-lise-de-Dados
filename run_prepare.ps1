[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PrepareArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$GamesCsv = Join-Path $ProjectRoot "data\raw\games.csv"
$RecommendationsCsv = Join-Path $ProjectRoot "data\raw\recommendations.csv"

if (-not (Test-Path $VenvPython)) {
    & (Join-Path $ProjectRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path $GamesCsv) -or -not (Test-Path $RecommendationsCsv)) {
    Write-Host "Coloque data\raw\games.csv e data\raw\recommendations.csv antes de preparar os dados." -ForegroundColor Yellow
    exit 1
}

& $VenvPython -m src.prepare_data @PrepareArgs
