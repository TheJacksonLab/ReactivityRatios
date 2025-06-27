# ReactivityRatios Utils Module

This directory contains utility modules that provide core functionality for the ReactivityRatios project, a comprehensive toolkit for calculating reactivity ratios in copolymerization reactions and generating transition state geometries.

## Overview

The utils module provides essential functions for:
- Chemical data manipulation and molecular conformational analysis
- File format conversion and validation
- Machine learning model training and evaluation
- SMARTS pattern manipulation and reaction processing
- Molecular force field calculations and geometry optimization

## Module Structure

### Core Modules

#### `chem_data.py`
**Chemical Data Processing and Analysis**

Primary classes and functions for molecular manipulation, conformation generation, and fingerprint analysis.

**Key Components:**
- `conformation` class: Manages molecular conformations with 3D coordinates
- `gen_conform()`: Generates optimized molecular conformations using MMFF force fields
- `smiles2graph()`: Converts SMILES to molecular graphs for visualization
- `get_morganFP()`: Computes Morgan fingerprints for molecular similarity analysis
- Various SMILES manipulation utilities (add/remove hydrogens, validation)

**Dependencies:** RDKit, NumPy, Matplotlib, PyTorch, py3Dmol

#### `model.py`
**Machine Learning Models and Training**

Comprehensive machine learning framework supporting multiple algorithms for both regression and classification tasks.

**Supported Models:**
- **XGBoost**: Gradient boosting for regression and classification
- **Neural Networks (ANN)**: PyTorch-based deep learning models
- **Linear Models**: Linear/Logistic regression
- **Support Vector Machines**: SVR and SVC
- **Decision Trees**: Regression and classification trees

**Key Features:**
- Automated model training with hyperparameter support
- Built-in evaluation metrics (RMSE, MAE, R², accuracy, F1-score)
- Batch processing and early stopping for neural networks
- Cross-platform GPU support (CUDA, MPS, CPU)

**Dependencies:** XGBoost, Scikit-learn, PyTorch, NumPy

#### `smarts_manipulation.py`
**SMARTS Pattern Processing and Reaction Analysis**

Specialized functions for processing reaction SMARTS patterns and molecular transformations.

**Core Functions:**
- `find_radical_atom_indices()`: Identifies radical centers in molecules
- `get_olefin_attack_idx()`: Determines olefin attack sites for polymerization
- `capping_smarts()`: Applies capping groups to reaction patterns
- `smarts2dicts()`: Converts SMARTS to reaction dictionaries with InChIKey mapping
- `substitute_smiles()`: Performs molecular substitution reactions
- `olefin_to_radical()`: Converts olefin structures to radical intermediates

**Dependencies:** RDKit, Pandas, NumPy

#### `molFF.py`
**Molecular Force Field Calculations and Geometry Optimization**

Advanced molecular mechanics and quantum chemistry interface for structure optimization.

**Key Functions:**
- `UFF_cons_opt()`: Universal Force Field optimization with constraints
- `smarts2prets_xyz()`: Generates pre-transition state geometries from SMARTS
- `smi_to_optimized_xyz()`: Full workflow from SMILES to optimized XYZ coordinates
- `do_smarts()`: Complete SMARTS processing pipeline with XTB optimization
- Geometric transformation utilities (rotation, translation)

**Dependencies:** RDKit, NumPy, XTB (external program)

#### `bel_trans.py`
**File Format Conversion Utility**

OpenBabel wrapper for converting between molecular file formats.

**Functionality:**
- Command-line interface for file conversion
- Support for major chemical file formats (SDF, MOL2, PDB, XYZ, etc.)
- Batch processing capabilities
- Format auto-detection

**Dependencies:** OpenBabel (Pybel)

#### `formatter.py`
**Data Validation and Format Checking**

CSV file validation with flexible column name matching.

**Features:**
- Case-sensitive/insensitive column validation
- Automatic column renaming for consistency
- Comprehensive error reporting
- Pandas DataFrame integration

**Dependencies:** Pandas

#### `descriptor_list.py`
**Molecular Descriptor Definitions**

Comprehensive lists of molecular descriptors for chemical analysis:
- **DFT descriptors**: Size/shape, electronic properties, surface properties, frontier molecular orbitals
- **RDKit descriptors**: 2D/3D molecular descriptors and functional group counts
- **Buried volume descriptors**: Steric environment analysis
- **General molecular information**: Basic molecular identifiers

## Usage Examples

### Basic Chemical Data Processing

```python
from utils.chem_data import gen_conform, smiles2graph, smile_addH

# Generate molecular conformations
smi = "CCO"  # Ethanol
conf = gen_conform(smi, smi_flag=True, conf_window=5)

# Add explicit hydrogens
smi_with_h = smile_addH("CCO")
print(smi_with_h)  # Output: [H]C([H])([H])C([H])([H])O[H]

# Visualize molecule
mol = smiles2graph("CCO", show=True)
```

### Machine Learning Model Training

```python
from utils.model import train_xgb_model_regressor, evaluate_xgb_model_regressor
import numpy as np

# Sample data
X_train = np.random.rand(100, 5)
y_train = np.random.rand(100)
X_test = np.random.rand(20, 5)
y_test = np.random.rand(20)

# Train XGBoost model
model = train_xgb_model_regressor(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
```

### SMARTS Processing

```python
from utils.smarts_manipulation import smarts2dicts, capping_smarts

# Process reaction SMARTS
smarts = "[CH2:1]=[CH:2][CH3:3].[CH3:4][CH2:5][*:6]>>[CH2:1][CH:2]([CH3:3])[CH2:5][CH3:4]"
inchikey_dict, reaction_dict = smarts2dicts(smarts)

# Apply capping groups
capped_smarts = capping_smarts(smarts, capping_frag='[*][H]')
```

### Geometry Optimization

```python
from utils.molFF import smi_to_optimized_xyz, do_smarts

# Optimize molecular geometry
smi_to_optimized_xyz("CCO", out_dir="./structures/")

# Process complete SMARTS reaction
smarts = "C[CH2*].C=C>>C[CH2]CC"
basename, sm_smiles = do_smarts(smarts, out_dir="./output/", prefix="TS")
```

## Installation Requirements

```bash
# Core dependencies
pip install rdkit numpy pandas matplotlib torch

# Optional dependencies for full functionality
pip install xgboost scikit-learn py3Dmol cairosvg

# External programs
# Install XTB from: https://github.com/grimme-lab/xtb
# Install OpenBabel from: https://openbabel.org/
```

## Configuration and Settings

The utilities are designed to work with the main project's settings system. Key configuration options include:

- **Computation settings**: Force field parameters, optimization criteria
- **Output formats**: File types and naming conventions  
- **Parallel processing**: Number of cores for model training
- **Chemical parameters**: Bond lengths, constraint forces for geometry optimization

## Integration with Main Project

These utilities integrate seamlessly with the main ReactivityRatios workflow:

1. **Data preprocessing**: `formatter.py` validates input CSV files
2. **Chemical analysis**: `chem_data.py` and `smarts_manipulation.py` process molecular data
3. **Geometry generation**: `molFF.py` creates optimized 3D structures
4. **Machine learning**: `model.py` trains predictive models for reactivity ratios
5. **File management**: `bel_trans.py` handles format conversions

## Contributing

When extending the utils module:

1. Follow the existing code structure and documentation patterns
2. Add comprehensive docstrings with parameter and return value descriptions
3. Include usage examples in docstrings for complex functions
4. Add new descriptor types to `descriptor_list.py` as needed
5. Ensure backward compatibility when modifying existing functions

## Performance Considerations

- **Large datasets**: Use batch processing in `model.py` for memory efficiency
- **Molecular conformations**: Adjust search parameters in `gen_conform()` based on molecular complexity
- **Parallel processing**: Leverage multi-core capabilities in XGBoost and neural network training
- **Memory management**: Be mindful of memory usage when processing large molecular datasets

## Troubleshooting

**Common Issues:**
- **RDKit sanitization errors**: Use `sanitize=False` in molecule creation for problematic structures
- **XTB not found**: Ensure XTB is installed and in system PATH
- **Memory errors in neural networks**: Reduce batch size or use CPU instead of GPU
- **File format issues**: Check OpenBabel installation for format conversion problems

For additional support, refer to the main project documentation or check the individual module docstrings for specific function usage.
