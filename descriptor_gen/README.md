# Molecular Descriptor Generation

Comprehensive pipeline for generating molecular descriptors from SMARTS patterns using quantum chemical calculations.

## Dependencies (Required External Software)
- **ORCA** - DFT calculations and electronic structure analysis
- **Multiwfn** - Wavefunction analysis for molecular properties
- **XTB** - Semi-empirical geometry optimization
- **Python**: RDKit, pandas, numpy, pyarrow, dbstep

## Workflow

### 1. Generate Geometries
```bash
python dimer_generate.py --settings settings/settings_Ccap.json
```
- Processes SMARTS patterns → XYZ coordinates
- XTB optimization for realistic geometries

### 2. Create ORCA Input Files  
```bash
python construct_orca_inp.py -xyz gen_xyz_cappingC -t orca_inp/wb97_cpcm_opt.inp -d DFT_cappingC
```
- Converts XYZ → ORCA input format
- Applies templates for different calculation types

### 3. Run ORCA Calculations (Manual)
```bash
cd DFT_cappingC
for inp in *.inp; do orca $inp > ${inp%.inp}.out & done; wait
```

### 4. Extract Descriptors
```bash
find DFT_cappingC -name "*.gbw" | xargs -I {} -P 8 python descriptor_gen.py --gbw_path {}
```
- Converts ORCA output → Molden format
- Runs Multiwfn for electronic properties
- Calculates RDKit descriptors and buried volumes

### 5. Compile Results
```bash
python parquet_compile.py --base_dir DFT_cappingC --output_file descriptors.parquet
```

## Key Scripts

- **`dimer_generate.py`** - SMARTS → 3D geometries  
- **`construct_orca_inp.py`** - XYZ → ORCA inputs
- **`descriptor_gen.py`** - ORCA output → molecular descriptors
- **`parquet_compile.py`** - JSON → ML-ready Parquet format
- **`example_run.sh`** - Complete pipeline example

## Configuration

Settings files in `settings/` control:
- Input/output directories
- Capping strategies (`[*]C`, `[*]H`, `[*]OC`)  
- Bond lengths for different reaction states
- Computational parameters

## Generated Descriptors

### DFT-Based (via Multiwfn)
- **Electronic**: HOMO/LUMO energies, orbital delocalization
- **Geometric**: Molecular size, planarity parameters
- **Surface**: Electrostatic potential, molecular volumes

### Chemical Informatics (via RDKit)
- **2D/3D descriptors**: LogP, TPSA, connectivity indices
- **Reactivity**: Buried volumes around reactive sites
- **Fingerprints**: Morgan fingerprints for similarity

## Output Format
- **JSON files** - Individual molecule descriptors
- **Parquet files** - Compiled ML-ready datasets with 100+ features per molecule

## Typical Usage
See `example_run.sh` for complete pipeline from SMARTS patterns to compiled descriptors.
