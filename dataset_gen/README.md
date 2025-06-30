# Dataset Generation for Reactivity Ratios

This directory contains tools and data for generating a structured dataset of polymer reactivity ratios from the Polymer Handbook using OCR processing and chemical compound recognition.

## Overview

The dataset generation pipeline processes OCR-extracted content from the Polymer Handbook to create a machine-readable dataset of reactivity ratio parameters. The main challenge is converting chemical compound names to standardized SMILES notation while validating their polymerization capability.

## Directory Structure

```
dataset_gen/
├── compound_recognizer.py      # Main processing script
├── setting.json               # API credentials configuration
├── dictable.pkl              # Cached chemical name-to-SMILES mappings
├── Handbook_wSMILES.csv      # Final processed dataset
├── Handbook_wSMILES.log      # Processing log of successful entries
├── Handbook_wSMILES_err.log  # Error log of unprocessed entries
├── OCR/                      # OCR processing results
│   ├── Handbook_res_filtered.md    # Filtered OCR input data
│   ├── Handbook_res.md            # Complete OCR results
│   ├── Reactivity_Ratio_Handbook/ # Source PDF files
│   ├── Handbook_res/              # Detailed OCR processing results
│   ├── filter.sh                  # OCR filtering script
│   └── README.md                  # OCR processing documentation
└── README.md                 # This file
```

## Key Files

### Core Processing Files

- **`compound_recognizer.py`** - Main script that processes OCR data and converts chemical names to SMILES notation. Uses multiple chemical identification services (CIRpy, Google Custom Search + PubChem) with intelligent fallback strategies.

- **`setting.json`** - Configuration file containing Google Custom Search API credentials needed for chemical compound identification.

- **`dictable.pkl`** - Persistent cache of chemical name-to-SMILES mappings to avoid redundant API calls and improve processing speed.

### Input/Output Files

- **`OCR/Handbook_res_filtered.md`** - Pre-filtered OCR-extracted table data from the Polymer Handbook serving as input for compound recognition.

- **`Handbook_wSMILES.csv`** - Final processed dataset containing reactivity ratio data with validated SMILES structures for both monomers.

### Log Files

- **`Handbook_wSMILES.log`** - Processing log containing successfully processed entries in pipe-delimited format.

- **`Handbook_wSMILES_err.log`** - Error log containing entries that could not be processed, useful for manual review and dataset improvement.

## Workflow

1. **OCR Processing** (in `OCR/` directory): Convert PDF handbook pages to markdown using Marker-PDF tool
2. **Data Filtering**: Filter OCR results to extract table rows with reactivity ratio information
3. **Compound Recognition**: Convert chemical names to SMILES using multiple identification services
4. **Structure Validation**: Verify that identified compounds are polymerizable monomers (contain C=C or C≡C bonds)
5. **Dataset Generation**: Output cleaned dataset with validated chemical structures

## Usage

```bash
# Ensure API credentials are configured in setting.json
python compound_recognizer.py
```

The script will:
- Process the filtered OCR data
- Identify and validate chemical compounds
- Generate the final dataset with SMILES annotations
- Cache results for future runs
- Log successful and failed processing attempts

## Chemical Compound Recognition Strategy

The compound recognition system uses a multi-tier approach:

1. **Cache Lookup**: Check local dictionary for previously identified compounds
2. **CIRpy Service**: Query Chemical Identifier Resolver for exact matches  
3. **Google Custom Search**: Use web search to find PubChem entries for fuzzy matching
4. **Structure Validation**: Verify compounds are polymerizable (contain reactive double/triple bonds)
5. **Substructure Validation**: Ensure compound derivatives contain expected base structures

## Data Quality Notes

Several factors affect dataset consistency:

1. **Web Search Variability**: Google search results can change over time, leading to different compound identifications across runs.

2. **Incomplete Processing**: Some chemical names cannot be automatically recognized and require manual curation (logged in error files).

3. **Filtering Differences**: Additional filtering steps (metal-containing compounds, fragmented molecules) may be applied downstream, causing discrepancies between raw processing logs and final datasets.

4. **OCR Errors**: Line breaks and formatting issues in OCR data are handled heuristically but may still cause processing failures.

## Future Improvements

- **Manual Curation**: Review entries in `Handbook_wSMILES_err.log` for manual SMILES annotation
- **Enhanced Validation**: Implement additional chemical structure validation rules
- **Alternative APIs**: Integrate additional chemical identification services for better coverage
- **Quality Control**: Add systematic validation of generated SMILES structures

## Dependencies

- `googleapiclient`: Google Custom Search API client
- `requests`: HTTP client for PubChem API access  
- `cirpy`: Chemical Identifier Resolver Python interface
- `rdkit`: Chemical informatics and structure validation
- `pandas`: Data manipulation and CSV output
- `tqdm`: Progress bar for long-running processing

## Source Citation

The processed data originates from the *Polymer Handbook*:

```bibtex
@book{brandrupPolymerHandbook1999,
  title = {Polymer Handbook},
  author = {Brandrup, J. and Immergut, E. H. and Grulke, Eric A.},
  year = {1999},
  edition = {4th},
  publisher = {Wiley},
  address = {New York},  
  isbn = {978-0-471-16628-3},
  chapter = {2},
  pages = {181--308},
  note = {Reactivity Ratios section}
}
```
