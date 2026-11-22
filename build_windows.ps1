$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$spec = Join-Path $PSScriptRoot 'BasiCAM_CNC.spec'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found. Run: py -m venv .venv"
}

& $python -m PyInstaller --noconfirm --clean $spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Built: $(Join-Path $PSScriptRoot 'dist\BasiCAM CNC.exe')"