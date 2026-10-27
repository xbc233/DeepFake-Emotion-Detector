param(
    [string]$InputVideo = "",
    [string]$Language = "zh"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Missing .venv Python. Follow README.md to install dependencies."
}
if (-not $InputVideo) {
    $InputVideo = Join-Path $projectRoot "data/raw/real/99b2678f1b838ad0361a1b578346f9d2.mp4"
}
$reportPath = Join-Path $projectRoot "data/results/demo_step1_report.json"
& $pythonPath (Join-Path $PSScriptRoot "run_pipeline.py") --input $InputVideo --language $Language --offline --output $reportPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host ""
Write-Host "Video seconds: $($report.video.duration_seconds)"
Write-Host "Run seconds: $($report.timings_seconds.total)"
Write-Host "Emotion mismatch score (not probability): $($report.deepfake_risk_score)"
Write-Host "Rule-based prediction: $($report.prediction)"
Write-Host "Report: $reportPath"
