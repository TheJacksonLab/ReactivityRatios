# ReactivityRatios: Machine Learning for Copolymer Reactivity Prediction

Repository for the submitted paper:

> **Chen J, Jackson N. Radical Reactivity Ratio Predictions for Copolymers with an Interpretable Machine Learning Model.** *ChemRxiv*. 2025; doi:10.26434/chemrxiv-2025-d2s45  
> *This content is a preprint and has not been peer-reviewed.*

## Overview

A computational toolkit for predicting copolymerization reactivity ratios using quantum chemical descriptors and machine learning. Combines DFT calculations, molecular descriptors, and interpretable ML models to predict how different monomers will copolymerize.

## Quick Start

### Dependencies
- **Python 3.8+** with RDKit, XGBoost, pandas, numpy
- **ORCA** - DFT calculations
- **Multiwfn** - Wavefunction analysis  
- **XTB** - Geometry optimization

### Basic Workflow
```bash
# 1. Generate molecular geometries from SMARTS patterns
cd descriptor_gen
python dimer_generate.py --settings settings/settings_Ccap.json

# 2. Run DFT calculations (requires ORCA)
python construct_orca_inp.py -xyz gen_xyz_cappingC -t orca_inp/wb97_cpcm_opt.inp -d DFT_cappingC
# (Run ORCA calculations manually)

# 3. Extract molecular descriptors
find DFT_cappingC -name "*.gbw" | xargs -I {} -P 8 python descriptor_gen.py --gbw_path {}
python parquet_compile.py --base_dir DFT_cappingC --output_file DFT_cappingC.parquet

# 4. Train ML models
cd ../ml_predict
python xgboost_MR_task3.py
```

## Directory Structure

- **`dataset/`** - Experimental reactivity ratio data and DFT-calculated molecular descriptors
- **`dataset_gen/`** - Tools for extracting data from polymer handbooks via OCR
- **`descriptor_gen/`** - Quantum chemical calculation pipeline (ORCA, Multiwfn, XTB)
- **`ml_predict/`** - Machine learning models for reactivity ratio prediction
- **`utils/`** - Core utilities for chemical data processing and ML
- **`DFT_barrier_result/`** - DFT energy barrier analysis results

## Key Features

- **Automated Dataset Generation**: OCR processing of polymer handbooks
- **Quantum Chemical Descriptors**: DFT-based molecular properties via ORCA/Multiwfn
- **Machine Learning Models**: XGBoost, neural networks, SVMs for prediction
- **Chemical Extrapolation**: Predict reactivity ratios for novel monomer pairs
- **Interpretable Models**: Feature importance analysis and SHAP values

## Data Sources

- **Polymer Handbook**: ~1,500 experimental reactivity ratio measurements
- **CKA Dataset**: High-quality manually curated data for validation
- **DFT Calculations**: 50,000+ quantum chemical calculations at multiple theory levels

## Citation

```bibtex
@article{chen2025radical,
  title={Radical Reactivity Ratio Predictions for Copolymers with an Interpretable Machine Learning Model},
  author={Chen, Jingdan and Jackson, Nicholas},
  journal={ChemRxiv},
  year={2025},
  doi={10.26434/chemrxiv-2025-d2s45}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
