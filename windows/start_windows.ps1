$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeRoot = Join-Path $ProjectRoot ".runtime"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "The PC environment is not installed. Double-click SETUP_PC.bat first."
}

$TesseractPathFile = Join-Path $RuntimeRoot "tesseract-path.txt"
if (Test-Path -LiteralPath $TesseractPathFile -PathType Leaf) {
    $Tesseract = (Get-Content -LiteralPath $TesseractPathFile -Raw).Trim()
    if (Test-Path -LiteralPath $Tesseract -PathType Leaf) {
        $env:AI_CAD_TESSERACT_CMD = $Tesseract
    }
}

$TessdataPathFile = Join-Path $RuntimeRoot "tessdata-path.txt"
if (Test-Path -LiteralPath $TessdataPathFile -PathType Leaf) {
    $Tessdata = (Get-Content -LiteralPath $TessdataPathFile -Raw).Trim()
    if (Test-Path -LiteralPath $Tessdata -PathType Container) {
        $env:TESSDATA_PREFIX = $Tessdata
    }
}

$DataRoot = Join-Path $ProjectRoot "data"
New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null
$env:AI_CAD_FEEDBACK_PATH = Join-Path $DataRoot "feedback.jsonl"
$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:HF_HUB_OFFLINE = "1"
$env:PYTHONUTF8 = "1"

Set-Location $ProjectRoot
Write-Host "AI CAD Converter is starting at http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "Keep this window open while using the program. Press Ctrl+C to stop it."
& $VenvPython app.py --server-name 127.0.0.1 --server-port 7860 --inbrowser
exit $LASTEXITCODE
