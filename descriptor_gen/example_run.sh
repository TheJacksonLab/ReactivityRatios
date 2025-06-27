#!/bin/bash
#$ -N example_run
#$ -cwd
#$ -j y
#$ -V
#$ -o n1_featurization.out
#$ -pe orte 48
#$ -q all.q


# step 1: generate xyz files
python dimer_generate.py --settings settings_CKAs.json 
python dimer_generate.py --settings textbook_settings_Ccap.json


# step 2: generate ORCA input files
python construct_orca_inp.py -xyz gen_xyz_cappingC -t orca_inp/b973c_spe.inp -d DFT_cappingC
python construct_orca_inp.py -xyz gen_xyz_CKAs_cappingC -t orca_inp/b973c_spe.inp -d DFT_CKAs_cappingC

# step 3: run DFT calculations with ORCA, omit here
# Note: This step is not included in the script, as it requires manual intervention to run

# step 4: generate descriptors with Multiwfn and RDKit
find DFT_CKAs_cappingC - maxdepth 2 -type f -name "*.gbw" | xargs -I {} -P 2 python descriptor_gen.py --gbw_path {} 
find DFT_cappingC -maxdepth 2 -type f -name "*.gbw" | xargs -I {} -P 32 python descriptor_gen.py --gbw_path {} 


# step 5: compile descriptors into parquet files
python parquet_compile.py --base_dir DFT_CKAs_cappingC --output_file DFT_CKAs_cappingC.parquet
python parquet_compile.py --base_dir DFT_cappingC --output_file DFT_cappingC.parquet
