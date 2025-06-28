# Molecular Descriptor Generation Pipeline

A comprehensive computational chemistry pipeline for generating molecular descriptors from reactivity ratio calculations. This toolkit processes monomer SMARTS patterns, generates dimer geometries, performs quantum chemical calculations, and extracts molecular descriptors using **ORCA**, **Multiwfn**, and **XTB**.

## Overview

This pipeline provides an end-to-end workflow for:
1. **Geometry Generation**: Create dimer geometries from SMARTS patterns
2. **Quantum Chemical Calculations**: Perform DFT calculations using ORCA
3. **Descriptor Extraction**: Generate molecular descriptors using Multiwfn and RDKit
4. **Data Compilation**: Compile results into machine learning-ready formats

## Dependencies

### ⚠️ Required Software (Must be installed separately)
- **ORCA**: Quantum chemistry package for DFT calculations
- **Multiwfn**: Wavefunction analysis program for extracting electronic properties  
- **XTB**: Semi-empirical quantum chemistry program for geometry optimization

### Python Dependencies
```bash
pip install rdkit-pypi pandas numpy pyarrow tqdm openbabel-wheel dbstep
```

### Installation Notes
1. **ORCA**: Download from [ORCA Forum](https://orcaforum.kofo.mpg.de/) and ensure `orca` and `orca_2mkl` are in your PATH
2. **Multiwfn**: Download from [http://sobereva.com/multiwfn/](http://sobereva.com/multiwfn/) and ensure `Multiwfn` is in your PATH
3. **XTB**: Install following instructions at [https://github.com/grimme-lab/xtb](https://github.com/grimme-lab/xtb)

## Directory Structure

```
descriptor_gen/
├── data/                           # Input data and lookup tables
│   ├── *.csv                      # Input CSV files with SMARTS patterns
│   └── *.pkl                      # Generated pickle files for mappings
├── settings/                       # Configuration files
│   ├── settings_*.json            # Different pipeline configurations
├── orca_inp/                      # ORCA input templates
│   ├── wb97_cpcm_opt.inp         # Optimization template
│   ├── b973c_spe.inp             # Single point energy template
│   └── *.inp                     # Other calculation templates
├── dimer_generate.py              # Step 1: Generate geometries from SMARTS
├── construct_orca_inp.py          # Step 2: Create ORCA input files
├── descriptor_gen.py              # Step 3: Extract descriptors from calculations
├── parquet_compile.py             # Step 4: Compile results for ML
└── example_run.sh                 # Complete pipeline example
```

## Workflow

### Step 1: Generate Geometries
```bash
python dimer_generate.py --settings settings/settings_Ccap.json
```
- Processes SMARTS patterns from CSV files
- Generates XYZ coordinate files for dimers using **XTB** optimization
- Creates InChI key mappings

### Step 2: Create ORCA Input Files
```bash
python construct_orca_inp.py -xyz gen_xyz_cappingC -t orca_inp/wb97_cpcm_opt.inp -d DFT_cappingC
```
- Converts XYZ files to **ORCA** input format
- Applies charge/multiplicity from Gaussian files
- Uses templates for different calculation types

### Step 3: Run ORCA Calculations
```bash
# Manual step - run ORCA calculations
cd DFT_cappingC
for inp in *.inp; do
    orca $inp > ${inp%.inp}.out &
done
wait
```
- Requires **ORCA** to be properly installed and licensed

### Step 4: Extract Descriptors
```bash
find DFT_cappingC -name "*.gbw" | xargs -I {} -P 8 python descriptor_gen.py --gbw_path {}
```
- Converts **ORCA** output to Molden format using `orca_2mkl`
- Runs **Multiwfn** for electronic property analysis
- Calculates RDKit descriptors and buried volumes
- Generates JSON files with all descriptors

### Step 5: Compile Results
```bash
python parquet_compile.py --base_dir DFT_cappingC --output_file descriptors.parquet
```
- Compiles all JSON files into Parquet format
- Handles large datasets efficiently
- Creates ML-ready feature matrices

## Configuration Files

### Settings JSON Format
```json
{
    "data_dir": "data/",
    "input_file": "monomer_react_ratio_smarts.csv",
    "output_dir": "gen_xyz_cappingC",
    "skip_geometry": false,
    "verbose": false,
    "bond_lengths": {
        "PreTS": 2.24,    // Transition state
        "": 1.8,          // Product
        "PreSm": 3.0      // Pre-reaction complex
    },
    "capping": "[*]C"     // Capping group for SMARTS
}
```

### Input CSV Format
Required columns:
- `Monomer_1`, `Monomer_2`: Monomer names
- `Monomer_1_smi`, `Monomer_2_smi`: SMILES strings
- `active_smi1`, `active_smi2`: Active site SMILES
- `r1`, `r2`: Reactivity ratios
- `smarts_11`, `smarts_12`, `smarts_21`, `smarts_22`: SMARTS reaction patterns

## Generated Descriptors

### DFT-Based Descriptors (via Multiwfn)
- **Basic Properties**: Molecular weight, atom count, charge, multiplicity
- **Geometric Properties**: Molecular size, radius, planarity parameters (MPP, SDP)
- **Electronic Properties**: Dipole/multipole moments, HOMO/LUMO energies
- **Orbital Delocalization**: Delocalization indices for frontier molecular orbitals
- **Surface Properties**: Electrostatic potential (ESP), ALIE, LEA analysis
- **Molecular Volume**: Overall and surface-specific volumes and areas

### RDKit Descriptors
- **2D Descriptors**: Molecular weight, LogP, TPSA, ring counts, etc.
- **3D Descriptors**: Molecular geometry-based properties
- **Fingerprints**: Morgan fingerprints for similarity analysis

### Reactive Site Descriptors
- **Buried Volume**: Steric accessibility around reactive centers (via dbstep)
- **Reactive Atom Identification**: Automatic detection of reaction sites

## Advanced Usage

### Custom ORCA Templates
Create new templates in `orca_inp/` directory:
```
! B3LYP def2-TZVP RIJCOSX Opt Freq
%pal nprocs 16 end
%maxcore 2500
*[XYZFILE]
```
The `[XYZFILE]` placeholder will be replaced with coordinates and charge/multiplicity.

### Parallel Processing
```bash
# Process multiple files in parallel
find DFT_directory -name "*.gbw" | xargs -I {} -P 16 python descriptor_gen.py --gbw_path {}
```

### Selective Descriptor Generation
```bash
# Skip certain descriptor types
python descriptor_gen.py --gbw_path file.gbw --no_BV_flag --no_fp_flag
```

### Custom Timeout Settings
```bash
# Set longer timeout for large molecules
python parquet_compile.py --base_dir DFT_dir --output_file results.parquet --timeout 120
```

## Output Files

### JSON Format (per molecule)
```json
{
    "AtomNum": 42,
    "Weight": 234.56,
    "HOMO": -0.234,
    "LUMO": -0.123,
    "Buried_Volume": 45.2,
    "smiles": "CC=C",
    "Morgan_Fingerprint": [0, 1, 0, ...],
    ...
}
```

### Parquet Format (compiled)
- Efficient columnar storage for large datasets
- Direct compatibility with pandas, scikit-learn
- Compressed format for reduced file sizes

## Troubleshooting

### Common Issues

1. **ORCA not found**: Ensure ORCA is in your PATH
   ```bash
   export PATH="/path/to/orca:$PATH"
   ```

2. **Multiwfn not found**: Set path manually in `descriptor_gen.py`:
   ```python
   def get_multiwfn_path():
       return "/path/to/Multiwfn"
   ```

3. **XTB optimization failures**: Check molecule validity and consider using different conformers

4. **Memory issues**: Adjust `%maxcore` in ORCA templates and reduce parallel processing

### Performance Tips

- Use SSD storage for intermediate files
- Optimize `%pal nprocs` in ORCA templates based on available cores
- Use appropriate `--timeout` values for large molecules
- Consider using ORCA's RIJCOSX approximation for faster SCF convergence

## Example Pipeline Execution

```bash
#!/bin/bash

# Step 1: Generate geometries
python dimer_generate.py --settings settings/settings_CKAs.json 
python dimer_generate.py --settings settings/textbook_settings_Ccap.json

# Step 2: Generate ORCA input files
python construct_orca_inp.py -xyz gen_xyz_cappingC -t orca_inp/b973c_spe.inp -d DFT_cappingC
python construct_orca_inp.py -xyz gen_xyz_CKAs_cappingC -t orca_inp/b973c_spe.inp -d DFT_CKAs_cappingC

# Step 3: Run DFT calculations (manual)
# Note: Run ORCA calculations manually or with job scheduler

# Step 4: Generate descriptors
find DFT_CKAs_cappingC -maxdepth 2 -type f -name "*.gbw" | xargs -I {} -P 2 python descriptor_gen.py --gbw_path {} 
find DFT_cappingC -maxdepth 2 -type f -name "*.gbw" | xargs -I {} -P 32 python descriptor_gen.py --gbw_path {} 

# Step 5: Compile descriptors
python parquet_compile.py --base_dir DFT_CKAs_cappingC --output_file DFT_CKAs_cappingC.parquet
python parquet_compile.py --base_dir DFT_cappingC --output_file DFT_cappingC.parquet
```

## Citation

If you use this pipeline in your research, please cite the relevant software:
- **ORCA**: Neese, F. (2012). *WIREs Comput. Mol. Sci.*, 2, 73-78. [DOI: 10.1002/wcms.81](https://doi.org/10.1002/wcms.81)
- **Multiwfn**: Lu, T.; Chen, F. (2012). *J. Comput. Chem.*, 33, 580-592. [DOI: 10.1002/jcc.22885](https://doi.org/10.1002/jcc.22885)
- **RDKit**: Landrum, G. et al. (2013). *J. Cheminform.*, 5, 25. [DOI: 10.1186/1758-2946-6-37](https://doi.org/10.1186/1758-2946-6-37)
- **XTB**: Bannwarth, C. et al. (2019). *WIREs Comput. Mol. Sci.*, 9, e1493. [DOI: 10.1002/wcms.1493](https://doi.org/10.1002/wcms.1493)

## License

This project is licensed under the MIT License.
