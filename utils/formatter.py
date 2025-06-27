import os
import pandas as pd
from typing import List, Optional


def csv_format_reader(expected_columns: List[str], filename: str, case_sensitive: bool = True) -> Optional[pd.DataFrame]:
    """
    Validate a CSV file format and column names. Returns the DataFrame if valid, None otherwise.
    
    Parameters:
    -----------
    expected_columns : List[str]
        List of expected column names in the CSV file
    filename : str
        Path to the CSV file to be validated
    case_sensitive : bool, default=True
        Flag to determine if column name validation should be case-sensitive
    
    Returns:
    --------
    Optional[pd.DataFrame]
        - The validated DataFrame if successful
        - None if any validation errors occur
    
    Example:
    --------
    >>> df = csv_format_val(['smiles', 'activity'], 'compounds.csv', False)
    >>> if df is not None:
    >>>     # Process the validated DataFrame
    >>>     print(f"Validation successful, dataframe shape: {df.shape}")
    >>> else:
    >>>     print("Validation failed")
    """
    # 1. Validate file extension and existence
    if not filename.lower().endswith('.csv'):
        print(f"Error: File '{filename}' does not have a .csv extension")
        return None
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist")
        return None
    
    # 2. Check if file is readable as CSV
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error: Could not read '{filename}' as CSV: {str(e)}")
        return None
    
    # 3. Validate column names
    actual_columns = list(df.columns)
    
    if not case_sensitive:
        # Convert to lowercase for case-insensitive comparison
        expected_lower = [col.lower() for col in expected_columns]
        actual_lower = [col.lower() for col in actual_columns]
        
        # Check if all expected columns are present (case-insensitive)
        missing_columns = [expected_columns[i] for i, col in enumerate(expected_lower) 
                          if col not in actual_lower]
        
        # If columns with different case exist, rename them to match expected case
        if not missing_columns:
            rename_dict = {}
            for i, exp_col in enumerate(expected_columns):
                try:
                    act_idx = actual_lower.index(exp_col.lower())
                    if actual_columns[act_idx] != exp_col:
                        rename_dict[actual_columns[act_idx]] = exp_col
                except ValueError:
                    pass
            
            if rename_dict:
                df = df.rename(columns=rename_dict)
                print(f"Renamed columns to match expected case: {rename_dict}")
    else:
        # Case-sensitive comparison
        missing_columns = [col for col in expected_columns if col not in actual_columns]
    
    if missing_columns:
        print(f"Error: Missing expected columns: {missing_columns}")
        return None
    
    return df
