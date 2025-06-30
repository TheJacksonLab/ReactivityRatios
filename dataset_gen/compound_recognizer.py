"""
Compound Recognizer for Reactivity Ratio Dataset Generation

This script processes OCR-extracted text from the Polymer Handbook to identify
chemical compounds and convert them to SMILES notation. It combines multiple
chemical identification services and validates monomer structures for 
polymerization capability.

Main workflow:
1. Parse tabular data from OCR markdown files
2. Identify chemical compounds using multiple services (CIRpy, Google Custom Search)
3. Validate SMILES structures for polymerizable monomers
4. Generate a cleaned dataset with SMILES annotations

Dependencies:
    - googleapiclient: Google Custom Search API
    - requests: HTTP requests for PubChem API
    - cirpy: Chemical Identifier Resolver
    - rdkit: Chemical structure analysis
    - pandas: Data handling
    - pickle: Caching chemical lookups
"""

import json
import os
import pickle
import re
from time import sleep
from urllib.parse import urlparse

import cirpy
import pandas as pd
import requests
from googleapiclient.discovery import build
from rdkit import Chem
from tqdm import tqdm

# Configuration and Constants
RE_PATTERN = r'^\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*$'
DF_COLUMNS = ['Monomer_1', 'Monomer_2', 'r1', 'r1_err', 'r2', 'r2_err', 'Conv', 'Refs', 'Monomer_1_smiles', 'Monomer_2_smiles']

# Load API credentials
with open('setting.json', 'r') as f:
    settings = json.load(f)
    MY_API_KEY = settings['my_api_key']
    MY_CSE_ID = settings['my_cse_id']

# Load or initialize chemical name-to-SMILES dictionary for caching
DICTABLE_FILE = 'dictable.pkl'
if os.path.exists(DICTABLE_FILE):
    with open(DICTABLE_FILE, 'rb') as f:
        dictable = pickle.load(f)
else:
    dictable = {}

# Initialize error log
open('Handbook_wSMILES_err.log', 'w').close()


def g_search(search_term, api_key, cse_id, **kwargs):
    """
    Perform Google Custom Search for chemical compounds.
    
    Args:
        search_term (str): Chemical name to search
        api_key (str): Google API key
        cse_id (str): Custom Search Engine ID
        **kwargs: Additional API parameters
        
    Returns:
        dict: JSON response from Google Custom Search API
    """
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res
    except Exception as e:
        print(f"Error during Google Custom Search: {search_term}")
        return {}


def lookup_name2smiles(name, dictable=dictable):
    """
    Look up cached SMILES for a chemical name.
    
    Args:
        name (str): Chemical name (whitespace will be normalized)
        dictable (dict): Cache dictionary
        
    Returns:
        str: SMILES string if found, empty string otherwise
    """
    normalized_name = name.strip().replace(' ', '')
    return dictable.get(normalized_name, '')


def get_smiles_from_pubchem_url(pubchem_url):
    """
    Extract SMILES from PubChem compound URL.
    
    Handles both CID numbers and compound names in URLs:
    - https://pubchem.ncbi.nlm.nih.gov/compound/123456 (CID)
    - https://pubchem.ncbi.nlm.nih.gov/compound/methane (name)
    
    Args:
        pubchem_url (str): PubChem compound URL
        
    Returns:
        str: SMILES string or empty string if not found
    """
    try:
        # Parse URL to extract compound identifier
        url_parts = urlparse(pubchem_url)
        path_parts = url_parts.path.strip('/').split('/')
        
        if len(path_parts) < 2 or path_parts[0] != 'compound':
            raise ValueError("Invalid PubChem URL format")
        
        compound_id = path_parts[1]
        
        # Build API URL based on identifier type
        if compound_id.isdigit():
            api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound_id}/property/CanonicalSMILES/JSON"
        else:
            api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_id}/property/CanonicalSMILES/JSON"
        
        # Fetch SMILES data
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        # Extract SMILES from response
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            return data['PropertyTable']['Properties'][0].get('CanonicalSMILES', '')
        
        return ''
        
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"Error extracting SMILES from {pubchem_url}: {e}")
        return ''


def google_name2smiles(name, api_key=MY_API_KEY, cse_id=MY_CSE_ID):
    """
    Find SMILES for chemical name using Google Custom Search + PubChem.
    
    Strategy: Search for the chemical name, check if top result is from PubChem,
    then extract SMILES from the PubChem page.
    
    Args:
        name (str): Chemical name to search
        api_key (str): Google API key
        cse_id (str): Custom Search Engine ID
        
    Returns:
        str: SMILES string or empty string if not found
    """
    search_results = g_search(search_term=name, api_key=api_key, cse_id=cse_id, num=1)
    
    if not search_results.get('items'):
        return ''
    
    result_url = search_results['items'][0]['link']
    if 'pubchem.ncbi.nlm.nih.gov' in result_url:
        return get_smiles_from_pubchem_url(result_url)
    
    return ''


def cirpy_name2smiles(compound_name):
    """
    Get SMILES using CIRpy (Chemical Identifier Resolver).
    
    Args:
        compound_name (str): Chemical name
        
    Returns:
        str: SMILES string or empty string if not found
    """
    try:
        smiles = cirpy.resolve(compound_name, 'smiles')
        return smiles if smiles else ''
    except Exception:
        return ''


def combined_name2smiles(compound_name, sleep_t=0.8, hard_flag=True):
    """
    Comprehensive chemical name to SMILES conversion using multiple services.
    
    Conversion strategy (in order):
    1. Check local cache (dictable)
    2. Try CIRpy for exact matches
    3. Try Google Custom Search for fuzzy matches (if hard_flag=True)
    
    Args:
        compound_name (str): Chemical name to convert
        sleep_t (float): Sleep time between API calls to avoid rate limiting
        hard_flag (bool): Whether to try Google search as fallback
        
    Returns:
        str: SMILES string or empty string if not found
    """
    if not compound_name or not compound_name.strip():
        return ''

    # Check cache first
    smiles = lookup_name2smiles(compound_name, dictable)
    if smiles:
        return smiles

    # Try CIRpy (exact match)
    smiles = cirpy_name2smiles(compound_name)
    sleep(sleep_t)  # Rate limiting
    
    if smiles:
        # Cache successful result
        dictable[compound_name.strip().replace(' ', '')] = smiles
        return smiles
    elif not hard_flag:
        return ''
    
    # Try Google Custom Search (fuzzy match)
    smiles = google_name2smiles(compound_name)
    sleep(sleep_t)  # Rate limiting
    
    if judge_mon_valid(smiles):
        # Cache only if the result is a valid monomer
        dictable[compound_name.strip().replace(' ', '')] = smiles
    
    return smiles


def is_substructure(smiles, smiles_sub):
    """
    Check if one molecule contains another as a substructure.
    
    Args:
        smiles (str): Main molecule SMILES
        smiles_sub (str): Substructure SMILES
        
    Returns:
        bool: True if smiles_sub is a substructure of smiles
    """
    mol = Chem.MolFromSmiles(smiles)
    mol_sub = Chem.MolFromSmiles(smiles_sub)
    
    if mol is None or mol_sub is None:
        return False
    
    return mol.HasSubstructMatch(mol_sub)


def judge_mon_valid(mon_smiles, sub_smiles=None):
    """
    Validate if a SMILES string represents a polymerizable monomer.
    
    Validation criteria:
    1. Must contain C=C or C≡C (polymerizable double/triple bond)
    2. If sub_smiles provided, must contain that substructure
    
    Args:
        mon_smiles (str): Monomer SMILES to validate
        sub_smiles (str, optional): Required substructure SMILES
        
    Returns:
        bool: True if valid polymerizable monomer
    """
    if not mon_smiles:
        return False
        
    mon_smiles = canonicalize_smiles(mon_smiles)
    if not mon_smiles:
        return False
    
    # Check for polymerizable bonds
    has_double_bond = is_substructure(mon_smiles, 'C=C')
    has_triple_bond = is_substructure(mon_smiles, 'C#C')
    
    if not (has_double_bond or has_triple_bond):
        return False
    
    # Check required substructure if provided
    if sub_smiles:
        sub_smiles = canonicalize_smiles(sub_smiles)
        if sub_smiles and not is_substructure(mon_smiles, sub_smiles):
            return False
    
    return True


def canonicalize_smiles(smiles):
    """
    Canonicalize SMILES string by removing charges and radicals.
    
    Normalization steps:
    1. Parse SMILES into molecule object
    2. Remove formal charges and radicals
    3. Generate canonical SMILES representation
    
    Args:
        smiles (str): Input SMILES string
        
    Returns:
        str: Canonicalized SMILES or empty string if invalid
    """
    if not smiles:
        return ''
        
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return ''
        
        # Normalize charges and radicals
        for atom in mol.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
        
        return Chem.MolToSmiles(mol)
        
    except Exception as e:
        print(f"Error canonicalizing SMILES '{smiles}': {e}")
        return ''


def judge_line_valid(line, pattern=RE_PATTERN):
    """
    Validate if a line matches the expected table format.
    
    Expected format: 8 data blocks separated by '|' characters
    Required blocks: Monomer_1, Monomer_2, r1, r2, References (non-empty)
    Numeric blocks: r1, r2, References (after cleaning)
    
    Args:
        line (str): Line to validate
        pattern (str): Regex pattern for table format
        
    Returns:
        tuple: (is_valid: bool, blocks: list)
    """
    match = re.match(pattern, line)
    if not match:
        return False, []
    
    blocks = [block.strip() for block in match.groups()]
    
    # Check block count
    if len(blocks) != 8:
        return False, []
    
    # Check required non-empty blocks: Monomer_1, Monomer_2, r1, r2, References
    required_indices = [0, 1, 2, 4, 7]
    if not all(blocks[i] for i in required_indices):
        return False, []
    
    # Check numeric blocks
    numeric_indices = [2, 4, 7]  # r1, r2, References
    for idx in numeric_indices:
        cleaned = blocks[idx].replace('.', '').replace('-', '')
        if not cleaned.isdigit():
            return False, []
    
    return True, blocks


def _log_error(line, log_file='Handbook_wSMILES_err.log'):
    """Log problematic lines for manual review."""
    with open(log_file, 'a') as f:
        f.write(line + '\n')


def _validate_monomer_pair(mon1, mon2, sleep_t):
    """
    Validate a pair of monomers and return their SMILES.
    
    For compound names like "Acetylene, phenyl-", extracts the base compound
    "Acetylene" as a required substructure.
    
    Args:
        mon1, mon2 (str): Monomer names
        sleep_t (float): Sleep time for API calls
        
    Returns:
        tuple: (mon1_smiles, mon2_smiles) or (None, None) if invalid
    """
    # Extract base compound names for substructure validation
    sub_name1 = mon1.split(', ')[0].strip() if ', ' in mon1 else None
    sub_name2 = mon2.split(', ')[0].strip() if ', ' in mon2 else None
    
    # Get SMILES for both monomers
    mon1_smiles = combined_name2smiles(mon1, sleep_t=sleep_t)
    mon2_smiles = combined_name2smiles(mon2, sleep_t=sleep_t)
    
    # Get substructure SMILES (without hard search to avoid excessive API calls)
    sub1_smiles = combined_name2smiles(sub_name1, hard_flag=False) if sub_name1 else None
    sub2_smiles = combined_name2smiles(sub_name2, hard_flag=False) if sub_name2 else None
    
    # Validate both monomers
    if (judge_mon_valid(mon1_smiles, sub1_smiles) and 
        judge_mon_valid(mon2_smiles, sub2_smiles)):
        return mon1_smiles, mon2_smiles
    
    return None, None


def _handle_line_break_error(line_num, content_lines, blocks, sleep_t):
    """
    Handle cases where monomer names are split across lines due to OCR errors.
    
    Strategy: Try combining current line with next line in different ways
    to reconstruct the original monomer names.
    
    Args:
        line_num (int): Current line number
        content_lines (list): All file lines
        blocks (list): Current line blocks
        sleep_t (float): Sleep time for API calls
        
    Returns:
        tuple: (success: bool, mon1_smiles: str, mon2_smiles: str)
    """
    if line_num + 1 >= len(content_lines):
        return False, None, None
    
    # Check if next line is also a valid table row
    next_line = content_lines[line_num + 1].strip()
    row_flag_next, _ = judge_line_valid(next_line)
    
    if row_flag_next:
        return False, None, None  # Next line is valid, current line is just invalid
    
    # Try to merge with next line
    blocks_next = next_line.split('|')
    if len(blocks_next) < 2:
        return False, None, None
    
    mon1, mon2 = blocks[0], blocks[1]
    next_mon1, next_mon2 = blocks_next[0].strip(), blocks_next[1].strip()
    
    # Try different combinations
    combinations = [
        (mon1 + next_mon1, mon2 + next_mon2),  # Direct concatenation
        (mon1 + next_mon2, mon2 + next_mon1),  # Swapped concatenation
    ]
    
    for test_mon1, test_mon2 in combinations:
        mon1_smiles, mon2_smiles = _validate_monomer_pair(test_mon1, test_mon2, sleep_t)
        if mon1_smiles and mon2_smiles:
            return True, mon1_smiles, mon2_smiles
    
    return False, None, None


def parse_table_file(file_path, pattern=RE_PATTERN, columns=DF_COLUMNS, sleep_t=0.8):
    """
    Parse OCR-extracted table file and convert to structured DataFrame.
    
    Processing steps:
    1. Validate each line against table format
    2. Extract monomer names and convert to SMILES
    3. Validate monomers for polymerization capability
    4. Handle OCR line-break errors
    5. Cache results and log progress
    
    Args:
        file_path (str): Path to input markdown file
        pattern (str): Regex pattern for table validation
        columns (list): DataFrame column names
        sleep_t (float): Sleep time between API calls
        
    Returns:
        pandas.DataFrame: Processed reactivity ratio data with SMILES
    """
    qualified_rows = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content_lines = file.readlines()

    # Initialize progress log
    open('Handbook_wSMILES.log', 'w').close()

    for line_num, line in tqdm(enumerate(content_lines), total=len(content_lines), desc="Processing lines"):
        # Periodic cache saving
        if line_num % 50 == 0:
            with open(DICTABLE_FILE, 'wb') as f:
                pickle.dump(dictable, f)
        
        try:
            row_flag, blocks = judge_line_valid(line.strip(), pattern)
            
            if not row_flag:
                _log_error(line.strip())
                continue
            
            mon1, mon2 = blocks[0], blocks[1]
            
            # Validate monomer pair
            mon1_smiles, mon2_smiles = _validate_monomer_pair(mon1, mon2, sleep_t)
            
            if mon1_smiles and mon2_smiles:
                # Success: add SMILES to blocks
                blocks.extend([mon1_smiles, mon2_smiles])
            else:
                # Try to handle line break errors
                success, mon1_smiles, mon2_smiles = _handle_line_break_error(
                    line_num, content_lines, blocks, sleep_t)
                
                if success:
                    blocks.extend([mon1_smiles, mon2_smiles])
                else:
                    _log_error(line.strip())
                    continue
            
            # Log successful processing
            qualified_rows.append(blocks)
            with open('Handbook_wSMILES.log', 'a') as f:
                f.write('|'.join(blocks) + '\n')
                
        except Exception as e:
            print(f"Error processing line {line_num}: {e}")
            _log_error(line.strip())
            continue
    
    return pd.DataFrame(qualified_rows, columns=columns)


if __name__ == "__main__":
    # Main processing pipeline
    input_file = 'OCR/Handbook_res_filtered.md'
    output_file = 'Handbook_wSMILES.csv'
    
    print("Starting compound recognition and SMILES conversion...")
    
    # Process the file
    df = parse_table_file(input_file, pattern=RE_PATTERN, 
                          columns=DF_COLUMNS, sleep_t=1.5)
    
    # Save results
    df.to_csv(output_file, index=False)
    
    # Save final cache
    with open(DICTABLE_FILE, 'wb') as f:
        pickle.dump(dictable, f)
    
    print(f"Processing complete! Results saved to '{output_file}'")
    print(f"Processed {len(df)} valid reactivity ratio entries")
