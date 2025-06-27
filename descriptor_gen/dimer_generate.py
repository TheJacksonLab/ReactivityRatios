#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dimer generation script for reactivity ratio calculations.

This script processes monomer SMARTS patterns, creates dictionaries for reactions,
and generates geometries for transition states and products using the XTB program.
All configuration is read from a settings.json file.
"""

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable rdkit warnings in force

import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm

# add ../ to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from utils.formatter import csv_format_reader
from utils.smarts_manipulation import smarts2dicts, capping_smarts
from utils.molFF import do_smarts
from utils.molFF import smi_to_optimized_xyz


def load_settings(settings_file):
    """
    Load settings from a JSON file.
    
    Parameters:
    -----------
    settings_file : str
        Path to the settings JSON file
        
    Returns:
    --------
    dict
        Dictionary containing all settings
    """
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        print(f"Error loading settings file: {str(e)}")
        return None


def main(settings_file):
    """
    Main function to process SMARTS patterns and generate dimer geometries.
    
    Parameters:
    -----------
    settings_file : str
        Path to the settings JSON file
    """
    # Load settings from JSON file
    settings = load_settings(settings_file)
    if settings is None:
        print("Failed to load settings. Exiting.")
        return
    
    # Extract configuration parameters
    data_dir = settings.get('data_dir', 'data/')
    monomer_rr_smarts_file = settings.get('input_file', 'monomer_react_ratio_smarts.csv')
    output_dir = settings.get('output_dir', 'gen_xyz')
    inchikey2smiles_file = settings.get('inchikey2smiles_file', 'inchikey2smiles_dic.pkl')
    inchikey_reaction_file = settings.get('inchikey_reaction_file', 'inchikey_reaction_dic.pkl')
    skip_geometry = settings.get('skip_geometry', False)
    verbose = settings.get('verbose', False)
    capping = settings.get('capping', '[*][H]') 
    
    # Bond lengths for different types of structures
    pre2CClength = settings.get('bond_lengths', {
        "PreTS": 2.24,  # Transition state
        "": 1.8,        # Product
        "PreSm": 3.0    # Pre-reaction complex
    })
    
    # Expected column names in the input CSV file
    column_names = settings.get('column_names', [
        "Monomer_1", "Monomer_1_smi", "active_smi1",
        "r1", "Monomer_2", "Monomer_2_smi", "active_smi2", "r2",
        "smarts_11", "smarts_12", "smarts_21", "smarts_22"
    ])
    
    # Dictionary files for storing reaction information
    inchikey2smiles_dic_file = os.path.join(data_dir, inchikey2smiles_file)
    inchikey_reaction_dic_file = os.path.join(data_dir, inchikey_reaction_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and validate the input CSV file
    input_csv = os.path.join(data_dir, monomer_rr_smarts_file)
    print(f"Reading input file: {input_csv}")
    worksheets = csv_format_reader(expected_columns=column_names, 
                                  filename=input_csv, 
                                  case_sensitive=False)
    
    if worksheets is None:
        print("Error: Failed to read or validate input CSV file.")
        return
    
    print(f"Successfully loaded data with {len(worksheets)} rows.")
    
    # Extract SMARTS patterns from the DataFrame
    smarts_columns = settings.get('smarts_columns', ['smarts_11', 'smarts_12', 'smarts_21', 'smarts_22'])
    smarts_all = worksheets[smarts_columns].values
    smarts_list = smarts_all.reshape(-1).tolist()
    
    # Add capping groups to SMARTS patterns
    # Capping groups (e.g., [*]C, [*]H) terminate dangling bonds for realistic geometries
    print("Processing SMARTS patterns...")
    smarts_list = list(map(lambda smarts: capping_smarts(smarts, capping_frag=capping), smarts_list))
    
    
    # Print a sample of the processed SMARTS patterns
    if verbose:
        print("\nSample of processed SMARTS patterns:")
        for i, smarts in enumerate(smarts_list[:min(5, len(smarts_list))]):
            print(f"smarts_{i+1}: {smarts}")
    
    # Create dictionaries for molecular identification and reaction mapping
    # These dictionaries enable lookup of molecules by InChI keys
    print("Creating reaction dictionaries...")
    inchikey2smiles_dic, inchikey_reaction_dic = {}, {}
    
    for smarts in smarts_list:
        # Convert each SMARTS pattern to molecule dictionaries
        # dic1: InChI key -> SMILES mapping
        # dic2: InChI key -> reaction information mapping
        dic1, dic2 = smarts2dicts(smarts)
        inchikey2smiles_dic.update(dic1)
        inchikey_reaction_dic.update(dic2)
    
    # Save dictionaries
    print(f"Saving dictionaries to {data_dir}")
    with open(inchikey2smiles_dic_file, "wb") as f:
        pickle.dump(inchikey2smiles_dic, f)
    
    with open(inchikey_reaction_dic_file, "wb") as f:
        pickle.dump(inchikey_reaction_dic, f)
    
    print(f"Dictionaries saved successfully!")
    
    # Generate 3D geometries if requested
    if not skip_geometry:
        print(f"Generating 3D geometries in {output_dir}...")
        inchikey2smiles_dict = {}
        
        # Create a progress bar for geometry generation
        # DEPENDENCY: Requires XTB for semi-empirical optimization
        for smarts in tqdm(smarts_list, desc="Generating geometries"):
            current_dir = os.getcwd()
            try:
                # Generate optimized geometries for individual reactants
                # Split reactants from products (>> separator)
                for smi in (smarts.split(">>")[0]).split("."):
                    smi_to_optimized_xyz(smi, out_dir=output_dir, verbose=verbose, gen_idx=True)
                
                # Generate constrained geometries for different reaction stages
                # Each stage has different C-C bond lengths to model reaction coordinate
                for key, value in pre2CClength.items():
                    inchikey, sm_smiles = do_smarts(
                        smarts, 
                        prefix=key,           # "PreTS", "", "PreSm" for different stages
                        blength=value,        # Bond length constraint
                        out_dir=output_dir,
                        verbose=verbose,
                        gen_idx=True,         # Generate atom index files
                    )
                    inchikey2smiles_dict[inchikey] = sm_smiles
            except Exception as e:
                print(f"Error processing SMARTS: {smarts}")
                print(f"Error details: {str(e)}")
            os.chdir(current_dir)
        
        # Save the mapping from InChI Keys to SMILES for generated structures
        # This file enables lookup of molecular structures by their unique identifiers
        inchikey_map_file = os.path.join(data_dir, "inchikey_smarts_map.pkl")
        with open(inchikey_map_file, "wb") as f:
            pickle.dump(inchikey2smiles_dict, f)
        
        print(f"Successfully generated geometries for {len(inchikey2smiles_dict)} structures.")
    
    print("Processing complete!")


if __name__ == "__main__":
    # Simple argument parser to get the settings file path
    parser = argparse.ArgumentParser(description="Generate dimer geometries from monomer SMARTS patterns using settings from JSON file")
    parser.add_argument("--settings", default="settings.json", help="Path to the settings JSON file")
    args = parser.parse_args()
    
    main(args.settings)
