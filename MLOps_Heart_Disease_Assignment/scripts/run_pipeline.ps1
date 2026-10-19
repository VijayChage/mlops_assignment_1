$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Step([string]$Title, [string]$Command, [string[]]$Arguments) {
    Write-Host "`n===== $Title =====" -ForegroundColor Cyan
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title (exit code $LASTEXITCODE)"
    }
}

Invoke-Step "Download UCI dataset" "python" @("scripts/download_data.py")
Invoke-Step "Generate EDA outputs" "python" @("scripts/eda.py")
Invoke-Step "Train, compare, and track models with MLflow" "python" @("scripts/train.py")
Invoke-Step "Run Ruff checks" "ruff" @("check", ".")
Invoke-Step "Run unit tests" "python" @("-m", "pytest", "-q")

Write-Host "`nPipeline completed successfully." -ForegroundColor Green
Write-Host "Model: artifacts/model.joblib"
Write-Host "Metadata/comparison: artifacts/model_metadata.json, artifacts/model_comparison.csv"
Write-Host "EDA/model plots: reports/figures"
Write-Host "MLflow database: mlflow.db"
