# Dataset Directory

Core datasets for copolymerization reactivity ratio prediction, including experimental data and DFT-calculated molecular descriptors.

## Key Files

### Experimental Data
- **`CKAs.csv`** - Manually curated high-quality reactivity ratio data for validation
- **`Handbook_original_filtered.csv`** - Experimental data from Polymer Handbook (4th edition)  
- **`monomer_react_ratio_smarts.csv`** - Main training dataset with SMARTS patterns
- **`textbook_monomer_react_ratio_smarts.csv`** - Curated textbook data with temperature documentation

### DFT Descriptors (Parquet format)
- **`DFT_cappingC.parquet`** - Methyl-capped descriptors (B97-3c level)
- **`DFT_CKAs_cappingC.parquet`** - High-quality CKA dataset descriptors
- **`wb97xd_DFT_cappingC.parquet`** - High-level theory descriptors (ωB97X-D/def2-TZVP)
- **`DFT_cappingH.parquet`** - Hydrogen-capped descriptors
- **`DFT_cappingOC.parquet`** - Methoxyl-capped descriptors

## Data Format

### Reactivity Ratios
- **r1, r2**: Reactivity ratios for monomer pairs
- Values > 1: preference for homopropagation
- Values < 1: preference for crosspropagation

### SMARTS Patterns  
- Format: `[radical].[monomer]>>[product]`
- Four patterns per monomer pair (smarts_11, smarts_12, smarts_21, smarts_22)
- Represent different reaction pathways

### Molecular Descriptors
Generated using ORCA, Multiwfn, and RDKit:
- **Electronic**: HOMO/LUMO energies, orbital delocalization indices
- **Geometric**: Molecular size, planarity, surface properties  
- **Chemical**: Dipole moments, electrostatic potential, buried volumes

## Usage

```python
import pandas as pd

# Load experimental data
data = pd.read_csv('monomer_react_ratio_smarts.csv')

# Load DFT descriptors  
descriptors = pd.read_parquet('DFT_cappingC.parquet')

# Typical filtering
filtered = data[(data['r1'] > 0.0) & (data['r1'] < 50) & 
               (data['r2'] > 0.0) & (data['r2'] < 50)]
```

## Statistics
- **~1,500 monomer pairs** across major vinyl monomer classes
- **200+ unique monomers** 
- **50,000+ DFT calculations**
- **100+ molecular descriptors** per molecule
