# Dataset Generation

Tools for generating structured datasets from polymer literature using OCR and automated chemical name recognition.

## Overview

Converts unstructured literature data into machine-readable formats for ML applications. Main challenge: chemical name standardization and SMILES validation.

## Key Components

### `compound_recognizer.py`
Main processing script with multi-tier chemical identification:
1. **Cache lookup** - Previously identified compounds
2. **CIRpy service** - Chemical Identifier Resolver
3. **Google Custom Search** - Web search for fuzzy matching
4. **Structure validation** - RDKit sanitization and polymerization capability checks

### Configuration & Data
- **`setting.json`** - API credentials (Google Custom Search, PubChem)
- **`dictable.pkl`** - Cached chemical name-to-SMILES mappings
- **`Handbook_wSMILES.csv`** - Final processed dataset with validated SMILES

### OCR Processing (`OCR/` directory)
- **`Reactivity_Ratio_Handbook/`** - Source PDF files
- **`Handbook_res_filtered.md`** - Pre-processed OCR input
- Uses Marker-PDF for PDF to markdown conversion

## Usage

```bash
# 1. Process PDFs with OCR (in OCR/ directory)
find Reactivity_Ratio_Handbook -name "*.pdf" | xargs -I {} marker_single {} --use_llm --output_dir Handbook_res

# 2. Configure API credentials in setting.json
# 3. Run compound recognition
python compound_recognizer.py

# 4. Followed data filtering
# ...
```

## Dependencies

- **Python**: RDKit, pandas, requests, cirpy, google-api-python-client
- **External**: Google Custom Search API, PubChem REST API
- **OCR**: Marker-PDF tool

## Limitations and Known Issues

**Dataset Inconsistencies**
- Filtering pipeline for non-reactivity entries, metal-containing compounds, and multi-fragment molecules is not fully automated
- Web search results vary over time, causing different compound identifications across processing runs
- Manual curation of failed identifications in error logs could improve dataset quality but was not implemented

**Recommended Improvements**
- Implement automated filtering for data quality control
- Cache search results to ensure reproducibility
- Develop manual curation workflow for unrecognized compounds
