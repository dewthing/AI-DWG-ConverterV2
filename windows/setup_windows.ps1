$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeRoot = Join-Path $ProjectRoot ".runtime"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-PythonCommand {
    if (Get-Command "py.exe" -ErrorAction SilentlyContinue) {
        return [PSCustomObject]@{ Command = "py.exe"; Prefix = @("-3") }
    }
    if (Get-Command "python.exe" -ErrorAction SilentlyContinue) {
        return [PSCustomObject]@{ Command = "python.exe"; Prefix = @() }
    }
    throw "Python 3.10 or newer was not found. Install 64-bit Python from https://www.python.org/downloads/windows/ and enable 'Add Python to PATH'."
}

function Get-TesseractPath {
    $OnPath = Get-Command "tesseract.exe" -ErrorAction SilentlyContinue
    if ($OnPath) {
        return $OnPath.Source
    }

    $Candidates = @()
    if (${env:ProgramFiles}) {
        $Candidates += Join-Path ${env:ProgramFiles} "Tesseract-OCR\tesseract.exe"
    }
    if (${env:ProgramFiles(x86)}) {
        $Candidates += Join-Path ${env:ProgramFiles(x86)} "Tesseract-OCR\tesseract.exe"
    }
    if ($env:LOCALAPPDATA) {
        $Candidates += Join-Path $env:LOCALAPPDATA "Programs\Tesseract-OCR\tesseract.exe"
    }
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }
    return $null
}

Set-Location $ProjectRoot
Write-Host "AI CAD Converter v6 - Windows PC setup" -ForegroundColor Green
Write-Host "Project: $ProjectRoot"

Write-Step "Checking Python"
$Python = Get-PythonCommand
$PythonVersion = & $Python.Command @($Python.Prefix) -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($LASTEXITCODE -ne 0) {
    throw "Python could not be started."
}
if ([version]$PythonVersion -lt [version]"3.10.0") {
    throw "Python $PythonVersion is too old. Install Python 3.10 or newer."
}
Write-Host "Python $PythonVersion"

Write-Step "Creating the private Python environment"
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    & $Python.Command @($Python.Prefix) -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv."
    }
}
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not update pip."
}
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install Python dependencies. Check the internet connection and run SETUP_PC.bat again."
}

Write-Step "Checking Tesseract OCR"
$Tesseract = Get-TesseractPath
if (-not $Tesseract) {
    $Winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
    if ($Winget) {
        Write-Host "Tesseract was not found. Installing the Windows package..."
        & $Winget.Source install --id UB-Mannheim.TesseractOCR --exact --source winget --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "winget could not install Tesseract automatically."
        }
        $Tesseract = Get-TesseractPath
    }
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
if ($Tesseract) {
    Set-Content -LiteralPath (Join-Path $RuntimeRoot "tesseract-path.txt") -Value $Tesseract -Encoding UTF8
    Write-Host "Tesseract: $Tesseract"

    $InstalledTessdata = Join-Path (Split-Path -Parent $Tesseract) "tessdata"
    $RuntimeTessdata = Join-Path $RuntimeRoot "tessdata"
    New-Item -ItemType Directory -Force -Path $RuntimeTessdata | Out-Null
    if (Test-Path -LiteralPath $InstalledTessdata -PathType Container) {
        Get-ChildItem -LiteralPath $InstalledTessdata -Filter "*.traineddata" -File | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $RuntimeTessdata -Force
        }
    }

    $ThaiData = Join-Path $RuntimeTessdata "tha.traineddata"
    if (-not (Test-Path -LiteralPath $ThaiData -PathType Leaf)) {
        Write-Host "Downloading official Thai OCR language data..."
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/tha.traineddata" -OutFile $ThaiData
        }
        catch {
            Write-Warning "Thai OCR data could not be downloaded. English OCR can still be used. Details: $($_.Exception.Message)"
            Remove-Item -LiteralPath $ThaiData -Force -ErrorAction SilentlyContinue
        }
    }
    if (Get-ChildItem -LiteralPath $RuntimeTessdata -Filter "*.traineddata" -File -ErrorAction SilentlyContinue) {
        Set-Content -LiteralPath (Join-Path $RuntimeRoot "tessdata-path.txt") -Value $RuntimeTessdata -Encoding UTF8
    }
}
else {
    Write-Warning "Tesseract is not installed. The converter will run, but editable OCR text will be unavailable until Tesseract is installed."
    Write-Host "Installer: https://github.com/UB-Mannheim/tesseract/wiki"
}

Write-Step "Verifying the application"
$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:HF_HUB_OFFLINE = "1"
if ($Tesseract) {
    $env:AI_CAD_TESSERACT_CMD = $Tesseract
}
$TessdataPathFile = Join-Path $RuntimeRoot "tessdata-path.txt"
if (Test-Path -LiteralPath $TessdataPathFile -PathType Leaf) {
    $env:TESSDATA_PREFIX = (Get-Content -LiteralPath $TessdataPathFile -Raw).Trim()
}
& $VenvPython -c "import cv2, ezdxf, fitz, gradio, pytesseract; from app import build_app; build_app(); print('Application check passed')"
if ($LASTEXITCODE -ne 0) {
    throw "The application check failed."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next time, double-click START_PC.bat to open the program."
