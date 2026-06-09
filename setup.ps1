[CmdletBinding()]
param(
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Invoke-BasePython {
    param([string[]]$Arguments)

    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        & $PythonCommand.Source @Arguments
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        return
    }

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & $PyLauncher.Source -3 @Arguments
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        return
    }

    throw "Python nao encontrado. Instale Python 3.11+ e tente novamente."
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $ProjectRoot "data\raw"), `
    (Join-Path $ProjectRoot "data\processed") | Out-Null

if (-not (Test-Path $VenvPython)) {
    Write-Host "Criando ambiente virtual em .venv..." -ForegroundColor Cyan
    Invoke-BasePython @("-m", "venv", ".venv")
}

if (-not $SkipInstall) {
    Write-Host "Instalando dependencias..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $VenvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "Ambiente pronto." -ForegroundColor Green
Write-Host "Coloque os arquivos abaixo antes de preparar os dados:" -ForegroundColor Yellow
Write-Host "  data\raw\games.csv"
Write-Host "  data\raw\recommendations.csv"
Write-Host "Depois rode: .\run.ps1"

