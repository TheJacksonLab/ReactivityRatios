# Utils Module

Core utility modules providing essential functionality for chemical data processing, machine learning, and molecular analysis.

## Overview

Provides reusable components for:
- Chemical data manipulation and molecular conformations
- Machine learning model training and evaluation  
- SMARTS pattern processing and reaction analysis
- Molecular force field calculations and geometry optimization
- File format conversion and data validation

## Key Modules

### `chem_data.py` - Chemical Data Processing
- **`conformation` class** - 3D molecular conformation management
- **`gen_conform()`** - MMFF force field optimization
- **`smiles2graph()`** - Molecular visualization
- **`get_morganFP()`** - Morgan fingerprint generation
- SMILES manipulation utilities

### `model.py` - Machine Learning Framework  
- **XGBoost** - Gradient boosting regression/classification
- **Neural Networks** - PyTorch-based deep learning
- **SVMs** - Support vector machines
- **Linear Models** - Ridge/Lasso regression
- Built-in evaluation metrics and cross-platform GPU support

### `smarts_manipulation.py` - Reaction Processing
- **`find_radical_atom_indices()`** - Identify reactive centers
- **`get_olefin_attack_idx()`** - Polymerization site detection
- **`capping_smarts()`** - Apply capping groups
- **`smarts2dicts()`** - Convert SMARTS to reaction dictionaries
- **`olefin_to_radical()`** - Radical intermediate generation

### `molFF.py` - Molecular Mechanics & QM Interface
- **`UFF_cons_opt()`** - Universal Force Field optimization
- **`smarts2prets_xyz()`** - Pre-transition state geometries
- **`smi_to_optimized_xyz()`** - SMILES → optimized coordinates
- **`do_smarts()`** - Complete SMARTS processing with XTB

### Supporting Modules
- **`bel_trans.py`** - OpenBabel format conversion
- **`formatter.py`** - CSV validation and formatting
- **`descriptor_list.py`** - Molecular descriptor definitions

## Usage Examples

```python
# Generate molecular conformations
from utils.chem_data import gen_conform
conf = gen_conform("CCO", smi_flag=True, conf_window=5)

# Train XGBoost model
from utils.model import train_xgb_model_regressor
model = train_xgb_model_regressor(X_train, y_train)

# Process SMARTS patterns  
from utils.smarts_manipulation import smarts2dicts
inchikey_dict, reaction_dict = smarts2dicts(smarts_pattern)

# Optimize molecular geometry
from utils.molFF import smi_to_optimized_xyz
smi_to_optimized_xyz("CCO", out_dir="./structures/")
```

## Dependencies

```bash
# Core scientific computing
pip install rdkit numpy pandas matplotlib torch

# Optional for full functionality  
pip install xgboost scikit-learn py3Dmol cairosvg

# External programs
# XTB: https://github.com/grimme-lab/xtb
# OpenBabel: https://openbabel.org/
```

## Integration

These utilities integrate throughout the ReactivityRatios workflow:
1. **Data preprocessing** - Validate and format input files
2. **Chemical analysis** - Process molecular structures and reactions
3. **Geometry generation** - Create optimized 3D coordinates
4. **Machine learning** - Train and evaluate predictive models
5. **File management** - Handle format conversions

## Descriptor Categories

- **DFT descriptors** - Electronic properties, orbital analysis, surface characteristics
- **RDKit descriptors** - 2D/3D molecular properties, connectivity indices
- **Buried volumes** - Steric environment analysis
- **Chemical informatics** - Fingerprints, similarity measures
