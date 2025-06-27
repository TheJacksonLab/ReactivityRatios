# Dataset Directory

This directory contains the core datasets used for reactivity ratio prediction in copolymerization systems. The files include experimental data, DFT-calculated molecular descriptors, and computational chemistry descriptors.

## File Descriptions

### Raw Data Files

**CKAs.csv**
- Contains reactivity ratio data for monomer pairs with copolymerization kinetic analysis (CKA). Collected manually from literatures.
- Includes monomer names, SMILES strings, active site representations, reactivity ratios (r1, r2), and SMARTS patterns
- Key columns: Monomer_1, Monomer_2, r1, r2, smarts_11, smarts_12, smarts_21, smarts_22, Temperature (K)
- Used for chemical extrapolation on machine learning models

**Handbook_original_filtered.csv**
- Experimental reactivity ratio data filtered from polymer handbooks records. (See ../dataset_gen/)
- Contains monomer pairs with measured reactivity ratios and associated uncertainties
- Key columns: Monomer_1, Monomer_2, r1, r1_err, r2, r2_err, following *Polymer Handbook* format
- Primary source of experimental validation data

**monomer_react_ratio_smarts.csv**
- Large dataset containing monomer reactivity ratios with SMARTS pattern representations derived from **Handbook_original_filtered.csv**
- Includes detailed molecular fingerprints and reaction mechanisms
- Used for ML model training

**textbook_monomer_react_ratio_smarts.csv**
- Curated dataset from textbook: *Principles of Polymerization* sources with SMARTS patterns
- Temperature documented compared with *Polymer Handbook*
- Used for benchmarking DFT calculated energy barrier performance on large-scale dataset

### DFT Calculation Results (Parquet Files)

**DFT_CKAs_cappingC.parquet**
- Methyl-capped DFT-calculated molecular descriptors for monomers in the CKA dataset
- Contains quantum chemical descriptors listed in SI: Table S2
- Computed under B97-3c

**DFT_cappingC.parquet**
- Methyl-capped DFT-calculated molecular descriptors for monomers in the *Polymer Handbook* dataset
- Contains quantum chemical descriptors listed in SI: Table S2
- Computed under B97-3c

**DFT_cappingC_cv.parquet**
- Methyl-capped DFT-calculated molecular descriptors for monomers in the *Polymer Handbook* dataset
- Charge variation (CV) on cation/anion wavefunction derived descriptor included
- Contains quantum chemical descriptors listed in SI: Table S2
- Computed under B97-3c

**DFT_cappingH.parquet**
- Hydrogen-capped DFT-calculated molecular descriptors for monomers in the *Polymer Handbook* dataset
- Contains quantum chemical descriptors listed in SI: Table S2
- Computed under B97-3c

**DFT_cappingOC.parquet**
- Methyoxyl-capped DFT-calculated molecular descriptors for monomers in the *Polymer Handbook* dataset
- Contains quantum chemical descriptors listed in SI: Table S2

**wb97xd_DFT_CKAs_cappingC.parquet**
- Methyl-capped DFT-calculated molecular descriptors for monomers in the CKAs dataset
- Contains quantum chemical descriptors listed in SI: Table S2
- Computed under RI-wB98X-D/def2-TZVP

**wb97xd_DFT_cappingC.parquet**
- Methyl-capped DFT-calculated molecular descriptors for monomers in the *Polymer Handbook* dataset
- Contains quantum chemical descriptors listed in SI: Table S2
- Computed under RI-wB98X-D/def2-TZVP

## Data Usage Notes

### Reactivity Ratios
- r1 and r2 represent the reactivity ratios for monomer pairs in copolymerization

### SMARTS Patterns
- Format: [radical].[monomer]>>[product], peoriodic connection to be capped is represented as `*`

### File Formats
- **CSV files**: Human-readable, suitable for data exploration and analysis
- **Parquet files**: Compressed binary format, optimized for large datasets and fast I/O
- Use pandas.read_parquet() for parquet files, pandas.read_csv() for CSV files


## Applications

These datasets support:
- Machine learning model development & benchmark for reactivity ratio prediction
- Quantum chemical descriptor analysis
- Copolymerization mechanism studies
- Chemical similarity and pattern analysis


