# Reactivity Ratio Calculation Toolkit

A comprehensive toolkit for calculating reactivity ratios in copolymerization reactions and generating transition state geometries.

## Overview

This toolkit provides a workflow for analyzing monomer reactivity ratios and generating transition state geometries for copolymerization reactions. It leverages RDKit for molecular manipulation and XTB for geometry optimization, offering a streamlined approach to studying polymerization kinetics.

## Features

- Process monomer SMARTS patterns from CSV data
- Generate dimer geometries for transition states and products
- Create comprehensive reaction dictionaries mapped by InChI keys
- Perform constrained geometry optimizations using XTB
- Configurable bond lengths for different reaction stages

## Directory Structure

```
└── Nick/
    ├── data/                   # Input data files
    │   └── monomer_react_ratio_smarts.csv  # Monomer data with SMARTS patterns
    ├── utils/                  # Utility modules
    │   ├── __init__.py         # Package initialization
    │   ├── chem_data.py        # Chemical data utilities
    │   ├── formatter.py        # CSV validation utilities
    │   ├── molFF.py            # Molecular force field calculations
    │   └── smarts_manipulation.py  # SMARTS pattern processing
    ├── gen_xyz/                # Generated geometry files (XYZ format)
    ├── dimer_generate.py       # Main script for dimer generation
    └── settings.json           # Configuration settings
```

## Getting Started

### Prerequisites

- Python 3.6+
- RDKit
- NumPy
- Pandas
- XTB (for geometry optimization)

### Installation

1. Clone this repository
2. Install required Python packages:
   ```
   pip install rdkit numpy pandas tqdm
   ```
3. Install XTB following the instructions at [https://github.com/grimme-lab/xtb](https://github.com/grimme-lab/xtb)

### Configuration

Edit the `settings.json` file to configure the toolkit:

```json
{
    "data_dir": "data/",
    "input_file": "monomer_react_ratio_smarts.csv",
    "output_dir": "gen_xyz",
    "skip_geometry": false,
    "verbose": false,
    
    "bond_lengths": {
        "PreTS": 2.24,
        "": 1.8,
        "PreSm": 3.0
    }
}
```

### Usage

Run the dimer generation script:

```bash
python dimer_generate.py
```

Or specify a custom settings file:

```bash
python dimer_generate.py --settings custom_settings.json
```

## File Formats

### Input CSV Format

The input CSV file should contain the following columns:
- Monomer_1: Name of first monomer
- Monomer_1_smi: SMILES string of first monomer
- active_smi1: Active site SMILES for first monomer
- r1: Reactivity ratio of first monomer
- Monomer_2: Name of second monomer
- Monomer_2_smi: SMILES string of second monomer
- active_smi2: Active site SMILES for second monomer
- r2: Reactivity ratio of second monomer
- smarts_11, smarts_12, smarts_21, smarts_22: SMARTS patterns for different reaction combinations

### Output Files

- XYZ geometry files in the output directory
- Pickle files containing:
  - InChI key to SMILES mapping
  - Reaction information dictionary
  - InChI key to SMARTS mapping

## Extending the Toolkit

The modular design allows for easy extension:

1. Add new utility functions in the utils directory
2. Update the settings.json file to include new parameters
3. Modify the scripts to incorporate new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RDKit team for the excellent cheminformatics toolkit
- XTB developers for the efficient semi-empirical quantum chemistry package
