#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ORCA Input File Generator

This script creates ORCA quantum chemistry input files from XYZ coordinate files
and templates. It handles charge/multiplicity extraction from Gaussian files and
supports charge variants for studying different oxidation states.

DEPENDENCY: Requires corresponding .gjf files with charge/multiplicity information.
"""

import pandas as pd 
import numpy as np
import os, sys
import argparse
import re

def get_cm_from_gjf(gjf_path):
    """
    Extract charge and multiplicity from a Gaussian input file (.gjf).
    
    This function reads a Gaussian format file and extracts the charge and
    multiplicity line, which is essential for setting up ORCA calculations
    with the correct electronic structure parameters.
    
    Parameters:
    -----------
    gjf_path : str
        Path to the Gaussian input file (.gjf)
        
    Returns:
    --------
    str or None
        String containing "charge multiplicity" or None if not found
        
    Note:
    -----
    The function looks for coordinate lines and extracts the charge/multiplicity
    line that appears immediately before the molecular coordinates.
    """
    if not os.path.isfile(gjf_path):
        print(f"File {gjf_path} does not exist")
        return None
    
    # Regular expression to match coordinate lines: Element X Y Z
    coord_re = r'^([A-Z][a-z]?)\s+([-]?\d*\.\d+|\d+)\s+([-]?\d*\.\d+|\d+)\s+([-]?\d*\.\d+|\d+)\s*$'
    
    with open(gjf_path, "r") as f:
        lines = f.readlines()
        
    result_i = None
    for i, line in enumerate(lines):
        if re.match(coord_re, line):
            result_i = i-1
            break
    if result_i is None:
        print(f"Could not find the coordinates in {gjf_path}")
        return None
    else:
        return lines[result_i].strip()
        

def change_template(xyz_path, template_path, dest_dir="./orca_int_rev", variant=0, variant_prefix=""):
    """
    Create ORCA input files by substituting coordinates into templates.
    
    This function takes XYZ coordinate files and ORCA input templates, then
    creates complete ORCA input files by:
    1. Reading charge/multiplicity from corresponding .gjf files
    2. Applying charge/multiplicity variants for different electronic states
    3. Substituting the [XYZFILE] placeholder with coordinates and electronic info
    
    Parameters:
    -----------
    xyz_path : str
        Path to the XYZ coordinate file
    template_path : str
        Path to the ORCA input template file containing [XYZFILE] placeholder
    dest_dir : str, optional
        Destination directory for generated ORCA input files (default: "./orca_int_rev")
    variant : int, optional
        Charge variant to apply: 0=neutral, +1=cation, -1=anion (default: 0)
    variant_prefix : str, optional
        Prefix to add to output filename (default: "")
        
    Returns:
    --------
    None
        Creates .inp files in the destination directory
        
    Note:
    -----
    The template file should contain [XYZFILE] placeholder which will be replaced
    with the molecular coordinates and charge/multiplicity information.
    
    DEPENDENCY: Requires corresponding .gjf files with charge/multiplicity info.
    """
    xyz_name = os.path.basename(xyz_path)
    gjf_path = os.path.join(os.path.dirname(xyz_path), f"{xyz_name.split('.')[0]}.gjf")
    if not os.path.isfile(gjf_path):
        print(f"File {gjf_path} does not exist")
        return
    template_name = os.path.basename(template_path)
    new_template_path = os.path.join(dest_dir, f"{variant_prefix}{xyz_name.split('.')[0]}.{template_name.split('.')[1]}")
    
    # Copy template file to destination with new name
    os.system(f"cp \"{template_path}\" \"{new_template_path}\"")
    
    # Extract and modify charge/multiplicity based on variant
    cm = get_cm_from_gjf(gjf_path)
    charge, multiplicity = list(map(int, cm.split()))
    
    # Apply charge variant (for studying different oxidation states)
    charge = charge + variant
    # Adjust multiplicity: odd/even electron count affects spin state
    multiplicity = (multiplicity + variant - 1) % 2 + 1
    cm = f"{charge} {multiplicity}"
    
    # Read XYZ coordinates (skip first two lines: atom count and comment)
    with open(xyz_path, "r") as f:
        xyz_content = f.readlines()
    xyz_content = "".join(xyz_content[2:])
    
    # Replace [XYZFILE] placeholder with molecular coordinates and electronic info
    # Format: xyz charge multiplicity
    #         atom1 x1 y1 z1
    #         atom2 x2 y2 z2
    #         ...
    #         *
    with open(new_template_path, "r") as f:
        lines = f.readlines()
    with open(new_template_path, "w") as f:
        for line in lines:
            f.write(line.replace("[XYZFILE]", f"xyz {cm} \n{xyz_content.strip()}\n*\n"))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Create ORCA input files from XYZ coordinates and templates.
    
    This script processes directories of XYZ files and creates ORCA input files
    by combining coordinate data with calculation templates. Charge and multiplicity
    information is extracted from corresponding Gaussian (.gjf) files.
    
    DEPENDENCY: Requires .gjf files with same basename as .xyz files for charge/multiplicity info.
    """)
    parser.add_argument("-xyz", "--xyz_path", type=str, required=True, 
                        help="Path to the xyz file directory (output directory of dimer_generate.py)")
    parser.add_argument("-t", "--template_path", type=str, required=True, 
                        help="Path to the ORCA input template file")
    parser.add_argument("-d", "--dest_dir", type=str, default="./orca_int_rev", 
                        help="Destination directory for the output .inp files")
    parser.add_argument("-v", "--variant", type=int, default=0, 
                        help="Charge variant: 0=neutral, +1=cation, -1=anion")
    
    args = parser.parse_args()
    xyz_path = args.xyz_path
    template_path = args.template_path
    dest_dir = args.dest_dir
    variant = args.variant
    
    # Validate input paths
    if not os.path.isdir(xyz_path) or not os.path.isfile(template_path):
        print("Error: Please check the path of the xyz directory or template file")
        sys.exit(1)
        
    # Find all XYZ files
    xyz_files = [os.path.join(xyz_path, f) for f in os.listdir(xyz_path) if f.endswith(".xyz")]
    
    if len(xyz_files) == 0:
        print("Error: No xyz files found in the directory")
        sys.exit(1)
        
    # Create destination directory
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        
    # Process each XYZ file
    print(f"Processing {len(xyz_files)} XYZ files...")
    for xyz_file in xyz_files:
        try:
            # Generate filename prefix based on charge variant
            if variant == 0:
                variant_prefix = ""
            elif variant > 0:
                variant_prefix = f"p{variant}_"
            else:
                variant_prefix = f"n{abs(variant)}_"
                
            change_template(xyz_file, template_path, dest_dir, variant=variant, variant_prefix=variant_prefix)
        except Exception as e:
            print(f"Error processing {xyz_file}: {e}")
            continue
            
    print(f"Successfully created ORCA input files in {dest_dir}")
    print("Done")
