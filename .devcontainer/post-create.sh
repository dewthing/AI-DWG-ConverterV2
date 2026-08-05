#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-tha
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

