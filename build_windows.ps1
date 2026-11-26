# Builds dist\BasiCAM CNC\ (one folder) with Nuitka; pass -Zip for the release archive, -Diagnostic for a console.
param([switch]$Zip, [switch]$Diagnostic)
$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found. Run: py -m venv .venv"
}

$arguments = @((Join-Path $PSScriptRoot 'build_nuitka.py'))
if ($Zip) { $arguments += '--zip' }
if ($Diagnostic) { $arguments += '--diagnostic' }
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed with exit code $LASTEXITCODE"
}
