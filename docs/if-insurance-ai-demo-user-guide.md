# IF Insurance AI Demo — User Guide

Version: 1.0

## Overview
The Insurance Claims Analyzer helps analyze claims and invoices against policy contracts and evaluate images with built‑in fraud detection.

- Claims document extraction and comparison
- Invoice support (PDF and image invoices)
- Fraud detection for documents and images
- Image damage analysis with false positive/negative scoring
- Exportable JSON reports

## Requirements
- Python 3.10+
- OpenAI API key (required)
- Anthropic API key (optional)
- Recommended: `pandoc` for Markdown → PDF export

## Setup
1) Install and run
```bash
cd /home/julian/dev/hobby/insurance-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```
2) Create a `.env` file
```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key   # optional
```

## Data Folder
Place documents and images under `data/` (subfolders supported):
- PDFs: claims, invoices, policy contracts
- Images: `.jpg`, `.jpeg`, `.png`

## Pages
### Claims Check
- Extract data from Claim, Invoice, Policy
- Compare side‑by‑side with discrepancies and recommendations
- Image invoices supported (analyze authenticity via Fraud Detection)

Workflow:
1. Add API key (sidebar or `.env`).
2. Use demo files or upload PDFs/images.
3. Extract Claim/Invoice/Policy.
4. Optional: include saved image analyses.
5. Compare to see mismatches and export report.

### Fraud Detection (inside Claims Check)
- Select any file from `data/` (PDF/image) via sidebar
- Detect doctored/fraudulent content and manipulation
- Output: risk score, authenticity, indicators, recommendations
- Export JSON report

### Image Detection
- Analysis modes: Comprehensive, Damage Assessment, Fraud Detection, False Positive/Negative, OCR
- Use demo images from `data/` or upload your own
- Save analyses for use in Claims Check
- Visualize severity/accuracy and export results

### Documentation
- Download this guide (Markdown/PDF)
- Steps to regenerate PDF locally

## API Keys
- OpenAI required; Anthropic optional
- Load from `.env` and/or override in sidebars

## Exports
- Claims comparison: JSON
- Image analysis: JSON or text
- Fraud detection: JSON
- User Guide: `docs/if-insurance-ai-demo-user-guide.(md|pdf)`

## Regenerate PDF
If `pandoc` is installed:
```bash
cd /home/julian/dev/hobby/insurance-demo
source .venv/bin/activate
pandoc docs/if-insurance-ai-demo-user-guide.md -o docs/if-insurance-ai-demo-user-guide.pdf
```
If missing (Debian/Ubuntu):
```bash
sudo apt-get update && sudo apt-get install -y pandoc
```

## Troubleshooting
- API key required: add via `.env` or sidebar
- PDF not generated: install `pandoc`
- Slow on large files: try smaller inputs
- Anthropic model without key: switch to OpenAI or add key

## Privacy & Security
- Avoid uploading sensitive data without proper agreements/consent
- Consider redaction of personal data

© IF Insurance AI Demo — For demonstration purposes only.
