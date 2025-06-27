#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Molecular Descriptor Compilation Pipeline

This script compiles molecular descriptors from individual JSON files into
a single Parquet file optimized for machine learning applications. It handles
large-scale descriptor datasets with parallel processing and timeout protection.

DEPENDENCIES:
- pyarrow: For efficient Parquet file operations
- pandas: For data manipulation
- numpy: For numerical operations

Input: Directory tree containing .molden.json files from descriptor_gen.py
Output: Single .parquet file with all molecular descriptors
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import functools
import time
import signal
from contextlib import contextmanager
import numpy as np

import argparse
from utils.descriptor_list import dft_descriptors, rdkit_descriptors, reac_BV_descriptors, general_mol_info

class TimeoutError(Exception):
    """Custom exception for timeout"""
    pass

@contextmanager
def time_limit(seconds):
    """
    Context manager for limiting execution time of a block of code
    
    Parameters:
    -----------
    seconds : int
        Maximum execution time in seconds
    """
    def signal_handler(signum, frame):
        raise TimeoutError(f"Process timed out after {seconds} seconds")
    
    # Set the signal handler and a alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def process_json_file(json_file, feature_names, timeout=30):
    """
    Process a single molden.json file and return its data with timeout
    
    Parameters:
    -----------
    json_file : str
        Path to the molden.json file
    feature_names : list
        List of feature names to extract
    timeout : int, optional
        Maximum processing time in seconds, defaults to 30 seconds
        
    Returns:
    --------
    tuple
        (directory_name, data_dict) where data_dict contains the processed data
        
    Raises:
    -------
    TimeoutError
        If processing takes longer than the specified timeout
    """
    # Extract directory name (molecule/system identifier)
    dir_name = Path(json_file).parent.name
    result = {}
    
    try:
        # Use the time_limit context manager to enforce timeout
        with time_limit(timeout):
            start_time = time.time()
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract only the features we're interested in
            for feature in feature_names:
                if feature in data:
                    # For Parquet, we need to flatten/simplify complex structures
                    if isinstance(data[feature], list):
                        # For arrays, handle based on size to optimize storage
                        # Large arrays (>100 elements) are converted to summary statistics
                        if len(data[feature]) > 100:  # Large arrays (e.g., fingerprints)
                            # Store summary statistics instead of full arrays
                            result[f"{feature}_mean"] = np.mean(data[feature]) if data[feature] else 0
                            result[f"{feature}_sum"] = sum(data[feature]) if data[feature] else 0
                            result[f"{feature}_nonzero"] = sum(1 for x in data[feature] if x != 0) if data[feature] else 0
                        else:
                            # Keep small arrays by flattening with prefix
                            for i, val in enumerate(data[feature]):
                                result[f"{feature}_{i}"] = val
                    elif isinstance(data[feature], dict):
                        # Flatten dictionaries with prefix
                        for k, v in data[feature].items():
                            if isinstance(v, (int, float, str, bool)):
                                result[f"{feature}_{k}"] = v
                    else:
                        # Simple types just keep as is
                        result[feature] = data[feature]
                else:
                    # If the feature is not present, set to None or a default value
                    result[feature] = None
            
            # Add molecule ID
            result['molecule_inchikey_id'] = dir_name
            
            processing_time = time.time() - start_time
            return (dir_name, result, processing_time)
            
    except TimeoutError:
        print(f"Timeout processing {json_file} (exceeded {timeout} seconds)")
        return (dir_name, {'molecule_id': dir_name}, timeout)
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return (dir_name, {'molecule_id': dir_name}, -1)  # -1 indicates error

def write_batch_to_parquet(output_file, data_batch, mode='w'):
    """
    Write a batch of data to a Parquet file
    
    Parameters:
    -----------
    output_file : str
        Path to output Parquet file
    data_batch : list
        List of dictionaries containing the data to write
    mode : str, optional
        Write mode, either 'w' (create new) or 'a' (append), defaults to 'w'
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data_batch)
        
        # Clean any NaN values
        df = df.fillna(0)
        
        if mode == 'w':
            # Write new file
            df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        else:
            # For append, we need to read existing file, concat, and write back
            try:
                existing_df = pd.read_parquet(output_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
            except Exception as e:
                print(f"Error appending data: {e}. Creating new file instead.")
                df.to_parquet(output_file, engine='pyarrow', compression='snappy')
            
        return True
    except Exception as e:
        print(f"Error writing to Parquet: {e}")
        return False

def compile_molden_data_to_parquet(base_dir, output_file, feature_names, timeout=30, max_workers=None):
    """
    Compile all molden.json files into a Parquet file for machine learning
    
    Parameters:
    -----------
    base_dir : str
        Base directory to search for molden.json files
    output_file : str
        Output Parquet file path
    feature_names : list
        List of feature names to extract
    timeout : int, optional
        Maximum processing time per file in seconds, defaults to 30 seconds
    max_workers : int, optional
        Maximum number of worker processes to use, defaults to number of CPU cores
    """
    # Find all molden.json files
    print(f"Searching for molden.json files in {base_dir}")
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            potential_file = os.path.join(root, dir_name, f"{dir_name}.molden.json")
            if os.path.exists(potential_file):
                json_files.append(potential_file)
    
    print(f"Found {len(json_files)} molden.json files")
    
    if not json_files:
        print("No files found. Exiting.")
        return
        
    print(f"Feature names: {feature_names}")
    
    # Process files in parallel
    timeout_count = 0
    error_count = 0
    success_count = 0
    total_processing_time = 0
    
    # Create a partial function with feature_names and timeout already set
    process_func = functools.partial(process_json_file, 
                                    feature_names=feature_names,
                                    timeout=timeout)
    
    # Use ProcessPoolExecutor for parallel processing with tqdm for progress
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Process all files
        results = list(tqdm(executor.map(process_func, json_files), 
                           total=len(json_files), 
                           desc="Processing files"))
        
        # Collect all results into a single list
        all_data = []
        
        for dir_name, data, processing_time in results:
            if processing_time == timeout:
                timeout_count += 1
            elif processing_time == -1:
                error_count += 1
            else:
                total_processing_time += processing_time
                success_count += 1
                
            if data:  # Skip empty results
                all_data.append(data)
        
        # Write all data at once
        if all_data:
            print(f"Writing all data ({len(all_data)} entries) to {output_file}")
            df = pd.DataFrame(all_data)
            df['react_atom_serial'] = df['react_atom_serial'].astype(str)
            df.to_parquet(output_file, index=False)
    
    # Print processing statistics
    print(f"\nProcessing Statistics:")
    print(f"  - Successfully processed: {success_count} files")
    print(f"  - Timed out: {timeout_count} files")
    print(f"  - Errors: {error_count} files")
    if success_count > 0:
        print(f"  - Average processing time: {total_processing_time / success_count:.2f} seconds per file")
    
    print(f"Successfully created Parquet file at {output_file}")
    
    # Print summary of Parquet file contents
    try:
        df = pd.read_parquet(output_file)
        print("\nParquet File Summary:")
        print(f"Total molecules: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        print("Feature statistics:")
        print(df.describe().T)
    except Exception as e:
        print(f"Error reading Parquet file for summary: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile molden.json files into Parquet')
    parser.add_argument('--base_dir', help='Base directory to search for molden.json files')
    parser.add_argument('--output_file', help='Output Parquet file path')
    parser.add_argument('--no_dft_descriptors', action='store_true', 
                        help='Include DFT descriptors')
    parser.add_argument('--no_rdkit_descriptors', action='store_true',
                        help='Include RDKit descriptors')
    parser.add_argument('--no_reac_BV_descriptors', action='store_true',
                        help='Include reactive Buried Volume descriptors')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout for processing each file in seconds')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of worker processes to use, default is CPU cores')

    
    args = parser.parse_args()
    
    all_descriptors = general_mol_info
    descriptors_to_include = ["general_mol_info"]
    if not args.no_dft_descriptors:
        all_descriptors.extend(dft_descriptors)
        descriptors_to_include.append("dft_descriptors")
    if not args.no_rdkit_descriptors:
        all_descriptors.extend(rdkit_descriptors)
        descriptors_to_include.append("rdkit_descriptors")
    if not args.no_reac_BV_descriptors:
        all_descriptors.extend(reac_BV_descriptors)
        descriptors_to_include.append("reac_BV_descriptors")
        
    print(f"Descriptors to include: {', '.join(descriptors_to_include)}")
    
    compile_molden_data_to_parquet(args.base_dir, args.output_file, all_descriptors, 
                                  timeout=args.timeout, max_workers=args.max_workers)
    print(f"Output file: {args.output_file}")