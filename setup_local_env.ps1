$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CacheDir = Join-Path $ProjectRoot ".pip-cache"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
$env:PIP_CACHE_DIR = $CacheDir

if (-not (Test-Path $VenvPython)) {
    py -3.12 -m venv .venv
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install --no-cache-dir -r requirements.txt
& $VenvPython -m ipykernel install --user --name orbit-wars-local --display-name "Python (Orbit Wars .venv)"

Write-Host ""
Write-Host "Done. In VS Code/Jupyter, select kernel: Python (Orbit Wars .venv)"
Write-Host "Python path: $VenvPython"
