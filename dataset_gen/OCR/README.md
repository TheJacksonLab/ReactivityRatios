# OCR Processing 

OCR extraction of reactivity ratio data from Polymer Handbook using Marker-PDF.

## Usage

```bash
# Process PDF files with LLM-enhanced OCR
find Reactivity_Ratio_Handbook -name "*.pdf" | xargs -I {} marker_single {} --use_llm --output_dir Handbook_res

# Filter results to extract reactivity ratio tables
./filter.sh
```

## Output
- **`Handbook_res/`** - Complete OCR processing results
- **`Handbook_res_filtered.md`** - Filtered data for downstream processing
- **`Reactivity_Ratio_Handbook/`** - Source PDF files

## Dependencies
- [Marker-PDF](https://github.com/datalab-to/marker) for OCR processing
- LLM access for enhanced accuracy

## Source Citation
Data from *Polymer Handbook* (4th Edition), Brandrup et al., Wiley, 1999.
