# OCR Processing with Marker-PDF

This directory contains OCR (Optical Character Recognition) processing results for the Polymer Handbook using the [Marker-PDF](https://github.com/datalab-to/marker) tool.

## Overview

Marker-PDF is a powerful tool that converts PDF documents to markdown format using advanced OCR and LLM capabilities. We use it here to extract text content from the Reactivity Ratio Handbook for further processing and analysis.

## Usage

To process PDF files in the Reactivity_Ratio_Handbook directory, run the following command:

```bash
find Reactivity_Ratio_Handbook -type f -name "*.pdf" | xargs -I {} marker_single {} --use_llm --output_dir Handbook_res
```

### Command Breakdown
- `find Reactivity_Ratio_Handbook -type f -name "*.pdf"` - Locates all PDF files in the handbook directory
- `xargs -I {}` - Passes each found file as an argument
- `marker_single {} --use_llm --output_dir Handbook_res` - Processes each PDF with LLM enhancement and saves to the output directory

## Output Structure

- **`./Handbook_res/`** - Contains the complete OCR processing results
- **`./Reactivity_Ratio_Handbook/`** - Source PDF files
- **`Handbook_res.md`** - Full recognition results in markdown format

**Note:** Only results for the first two pages are shown in the local directories. The complete processing results can be found in `Handbook_res.md`.

## Source Citation

The processed content originates from the *Polymer Handbook*:

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

## Requirements

- [Marker-PDF](https://github.com/datalab-to/marker) installed and configured
- Access to LLM services (when using `--use_llm` flag)
- Sufficient disk space for output files

