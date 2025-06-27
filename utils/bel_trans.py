#!/usr/bin/env python

import argparse
from openbabel.pybel import *
import sys

def convert_molecule_file(input_file, output_format, input_format=None):
    """
    Convert a molecule file to a different format using OpenBabel's Pybel.
    
    Parameters:
    -----------
    input_file : str
        Path to the input file
    output_format : str
        Desired output file format (e.g., 'sdf', 'mol2', 'pdb')
    input_format : str, optional
        Input file format. If not provided, it will be inferred from the input filename
        
    Returns:
    --------
    str
        Path to the output file
    """
    # Handle relative paths
    if "/" not in input_file:
        input_file = "./" + input_file
    
    # Split the path and filename
    path, ff_name = "/".join(input_file.split('/')[:-1]), input_file.split('/')[-1]
    basename, in_format = "_".join(ff_name.split('.')[:-1]), ff_name.split('.')[-1]
    
    # Use provided input format if specified
    if input_format:
        in_format = input_format
    
    # Ensure path ends with a slash
    if path and path[-1] != '/':
        path += '/'
    
    # Construct output file path
    output_file = path + basename + '.' + output_format
    
    # Convert file
    for mymol in readfile(in_format, input_file):
        mymol.write(output_format, output_file, overwrite=True)
    
    return output_file

def main():
    """Command line interface for the file conversion function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Using pybel to convert molecule files')
    
    try:
        parser.add_argument('input_file', type=str, help='filename of input')
        parser.add_argument('output_format', type=str, help='output file type')
        parser.add_argument('--inp', type=str, required=False, help='override input file format')
    except:
        print("Please confirm your input!")
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Call the conversion function
    output_file = convert_molecule_file(args.input_file, args.output_format, args.inp)
    print(f'Convert succeed! Output file: {output_file}')

if __name__ == "__main__":
    main()