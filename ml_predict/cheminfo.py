import pandas as pd
import numpy as np
import os, sys
from glob import glob
import pickle
import rdkit
from rdkit import Chem
from utils.descriptor_list import dft_els_prop, dft_FMO, dft_surface, dft_size_shape
from utils.descriptor_list import rdkit_descriptors, reac_BV_descriptors
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
from utils.smarts_manipulation import get_smi_prod
from joblib import Parallel, delayed
from math import sqrt

def FMO_trans(series):
    features = ['HOMO_a', 'HOMO_b', 'LUMO_a', 'LUMO_b', 'ODI_HOMO_a', 'ODI_HOMO_b', 'ODI_LUMO_a', 'ODI_LUMO_b',
        'HOMO', 'HOMO_1', 'LUMO', 'LUMO_Add1', 'ODI_HOMO', 'ODI_HOMO_1', 'ODI_LUMO', 'ODI_LUMO_Add1']
    
    for feature in features:
        if feature not in series:
            raise ValueError(f"{feature} not in given series data")
    
    if pd.isna(series['HOMO']):
        # this is an open-shell case
        if series['HOMO_a'] > series['HOMO_b']:
            series['HOMO'] = series['HOMO_a']
            series['HOMO_1'] = series['HOMO_b']
            series['ODI_HOMO'] = series['ODI_HOMO_a']
            series['ODI_HOMO_1'] = series['ODI_HOMO_b']
        else:
            series['HOMO'] = series['HOMO_b']
            series['HOMO_1'] = series['HOMO_a']
            series['ODI_HOMO'] = series['ODI_HOMO_b']
            series['ODI_HOMO_1'] = series['ODI_HOMO_a']
        
        if series['LUMO_a'] < series['LUMO_b']:
            series['LUMO'] = series['LUMO_a']
            series['LUMO_Add1'] = series['LUMO_b']
            series['ODI_LUMO'] = series['ODI_LUMO_a']
            series['ODI_LUMO_Add1'] = series['ODI_LUMO_b']
        else:
            series['LUMO'] = series['LUMO_b']
            series['LUMO_Add1'] = series['LUMO_a']
            series['ODI_LUMO'] = series['ODI_LUMO_b']
            series['ODI_LUMO_Add1'] = series['ODI_LUMO_a']
    
    return series

def search_inchikey(inchikey, worksheet, type_='int'):
    assert type_ in ['int', 'sm', 'ts']
    if type_=='int':
        prefix = ''
    elif type_=='sm':
        prefix = 'PreSm_'
    elif type_=='ts':
        prefix = 'PreTS_'
    assert "molecule_inchikey_id" in worksheet.columns, f"molecule_inchikey_id not in the worksheet"
    
    inchikey = prefix + inchikey
    if inchikey not in worksheet['molecule_inchikey_id'].values:
        return []
    
    res_inckikey = []

    if inchikey in worksheet['molecule_inchikey_id'].values:
        res_inckikey.append(inchikey)
        
    return res_inckikey

def get_discriptor(worksheet, inchikey, feature_list=[]):
    assert "molecule_inchikey_id" in worksheet.columns, f"molecule_inchikey_id not in the worksheet"
    if inchikey not in worksheet['molecule_inchikey_id'].values:
        return None
    
    # Filter worksheet to get just the target molecule
    molecule_data = worksheet[worksheet['molecule_inchikey_id'] == inchikey]
    
    # Filter columns based on feature list
    selected_columns = [col for col in feature_list if col in molecule_data.columns]
    
    return molecule_data[selected_columns].iloc[0]

def search_smiles(smiles, worksheet, type_='int', val_set=None):
    inchikey1 = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles, sanitize=True))
    res_lis1 = search_inchikey(inchikey1, worksheet, type_)
    if (len(res_lis1) > 0) and ((val_set is None) or (inchikey1 in val_set)):
        return res_lis1
    
    inchikey2 = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles, sanitize=False))
    res_lis2 = search_inchikey(inchikey2, worksheet, type_)
    if (len(res_lis2) > 0) and ((val_set is None) or (inchikey2 in val_set)):
        return res_lis2
    return []

def process_row(idx, exp_worksheet, mol_descriptors_rev, feature_list, capping_group, expected_mol_num, 
                r_flag=True, sm_flag=False, ts_flag=False):
    series = exp_worksheet.iloc[idx]
    mol1_name = series['Monomer_1']
    mol2_name = series['Monomer_2']
    r1 = series['r1']
    r2 = series['r2']
    
    mol1s = series['Monomer_1_smi']
    mol2s = series['Monomer_2_smi']
    mol1_inchiks = search_smiles(mol1s, mol_descriptors_rev, 'int')
    mol2_inchiks = search_smiles(mol2s, mol_descriptors_rev, 'int')

    if ts_flag:
        ts11s = get_smi_prod(series['smarts_11'].split(">>")[0], capping_frag=capping_group)
        ts12s = get_smi_prod(series['smarts_12'].split(">>")[0], capping_frag=capping_group)
        ts21s = get_smi_prod(series['smarts_21'].split(">>")[0], capping_frag=capping_group)
        ts22s = get_smi_prod(series['smarts_22'].split(">>")[0], capping_frag=capping_group)
        ts11_inchiks = search_smiles(ts11s, mol_descriptors_rev, 'ts')
        ts12_inchiks = search_smiles(ts12s, mol_descriptors_rev, 'ts')
        ts21_inchiks = search_smiles(ts21s, mol_descriptors_rev, 'ts')
        ts22_inchiks = search_smiles(ts22s, mol_descriptors_rev, 'ts')
    else:
        ts11_inchiks = []
        ts12_inchiks = []
        ts21_inchiks = []
        ts22_inchiks = []

    if sm_flag:
        sm11s = get_smi_prod(series['smarts_11'].split(">>")[0], capping_frag=capping_group)
        sm12s = get_smi_prod(series['smarts_12'].split(">>")[0], capping_frag=capping_group)
        sm21s = get_smi_prod(series['smarts_21'].split(">>")[0], capping_frag=capping_group)
        sm22s = get_smi_prod(series['smarts_22'].split(">>")[0], capping_frag=capping_group)
        sm11_inchiks = search_smiles(sm11s, mol_descriptors_rev, 'sm')
        sm12_inchiks = search_smiles(sm12s, mol_descriptors_rev, 'sm')
        sm21_inchiks = search_smiles(sm21s, mol_descriptors_rev, 'sm')
        sm22_inchiks = search_smiles(sm22s, mol_descriptors_rev, 'sm')
    else:
        sm11_inchiks = []
        sm12_inchiks = []
        sm21_inchiks = []
        sm22_inchiks = []

    if r_flag:
        rad1s = get_smi_prod(series['active_smi1'], capping_frag=capping_group)
        rad2s = get_smi_prod(series['active_smi2'], capping_frag=capping_group)
        rad1_inchiks = search_smiles(rad1s, mol_descriptors_rev, 'int')
        rad2_inchiks = search_smiles(rad2s, mol_descriptors_rev, 'int')
    else:
        rad1_inchiks = []
        rad2_inchiks = []
    
    # Alphabetically reorder the monomer1 and monomer2 columns
    if mol1_name > mol2_name:
        mol1_name, mol2_name = mol2_name, mol1_name
        r1, r2 = r2, r1
        mol1s, mol2s = mol2s, mol1s
        mol1_inchiks, mol2_inchiks = mol2_inchiks, mol1_inchiks

        rad1_inchiks, rad2_inchiks = rad2_inchiks, rad1_inchiks

        ts11_inchiks, ts12_inchiks = ts22_inchiks, ts21_inchiks
        ts21_inchiks, ts22_inchiks = ts12_inchiks, ts11_inchiks

        sm11_inchiks, sm12_inchiks = sm22_inchiks, sm21_inchiks
        sm21_inchiks, sm22_inchiks = sm12_inchiks, sm11_inchiks
    
    potential_inchiks = [mol1_inchiks, mol2_inchiks, rad1_inchiks, rad2_inchiks, ts11_inchiks, ts12_inchiks, ts21_inchiks, ts22_inchiks,
                         sm11_inchiks, sm12_inchiks, sm21_inchiks, sm22_inchiks]
    potential_prefix = ['mol1_', 'mol2_', 'rad1_', 'rad2_', 'ts11_', 'ts12_', 'ts21_', 'ts22_',
                        'sm11_', 'sm12_', 'sm21_', 'sm22_']

    final_inchiks = []
    final_prefix = []
    for inchik_list, prefix in zip(potential_inchiks, potential_prefix):
        if len(inchik_list) > 0:
            final_inchiks = final_inchiks + inchik_list
            final_prefix = final_prefix + [prefix]
    
    res_features = pd.Series([mol1_name, mol2_name, r1, r2],
                          index=['Monomer_1', 'Monomer_2', 'r1', 'r2'])
    
    try:
        assert len(final_inchiks) == expected_mol_num, \
            f"len of inchikeys not equal to {expected_mol_num}, {final_inchiks}"
        for (prefix, inchik) in zip(final_prefix, final_inchiks):
            des_line = get_discriptor(mol_descriptors_rev, inchik, feature_list=feature_list)
            # change the column name
            des_line.index = [prefix + i for i in des_line.index]
            
            # concat 2 series
            res_features = pd.concat([res_features, des_line], axis=0)
        
        return res_features
    except:
        return None

def gen_feature_df_paral(exp_worksheet, mol_descriptors_rev, feature_list, capping_group="[*]C", expected_mol_num=4,
                         r_flag=True, sm_flag=False, ts_flag=False):
    
    # Assuming these variables are defined elsewhere or should be passed as arguments
    # exp_worksheet, mol_descriptors_rev, feature_list
    
    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # List to store results
    dataframe_value = []
    error_count = 0
    success_count = 0
    
    # Create a list of indices to process
    indices = list(range(exp_worksheet.shape[0]))
    


    results = Parallel(n_jobs=num_cores)(
        delayed(process_row)(
            idx, exp_worksheet=exp_worksheet, mol_descriptors_rev=mol_descriptors_rev,
            feature_list=feature_list, capping_group=capping_group, expected_mol_num=expected_mol_num,
            r_flag=r_flag, sm_flag=sm_flag, ts_flag=ts_flag
        ) for idx in tqdm(indices)
    )

    # for MACOS
    # results = []
    # for idx in tqdm(range(exp_worksheet.shape[0])):
    #     result = process_func(idx)
    #     results.append(result)
    
    # Process results
    for result in results:
        if result is not None:
            dataframe_value.append(result)
            success_count += 1
        else:
            error_count += 1
    
    print(f"num of success: {success_count}")
    print(f"num of error: {error_count}")
    
    # Combine all results into a single DataFrame if needed
    if dataframe_value:
        # final_df = pd.concat(dataframe_value, axis=1, keys=range(len(dataframe_value))).T
        final_df = pd.DataFrame(dataframe_value)
        return final_df
    else:
        return pd.DataFrame()


def judge_type(x_tuple, d=0.5):
    r1 = x_tuple[0]
    r2 = x_tuple[1]
    if len(x_tuple) == 2:
        r1_r2 = r1*r2
    else:
        r1_r2 = x_tuple[2]
    
    if (r1**2 + r2**2) < 2*(d**2) or (r1 < 0 and r2 < 0):
        return 0 # 'type 0'
    elif (r1 - 1)**2 + (r2 - 1)**2 < (d**2):
        return 1 # 'type 1'
    elif r1_r2 < (1+d/sqrt(2))**2 and r1_r2 > (1-d/sqrt(2))**2:
        return 2 # 'type 2'
    elif r1_r2 > 1:
        return 4 # 'type 4', merged 2 types: blocky & pseudo-blocky
    elif r1_r2 < 1:
        return 3 # 'type 3'
    else:
        return 4 # 'type 4'
    
def unpack_list_column(df, column_name='r'):
    """
    Unpacks a DataFrame column containing lists into multiple rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the list column to unpack
    column_name : List or str, default='r'
        The name of the column containing lists
        
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the list column unpacked
    """
    # Create a list to store the new rows
    new_rows = []
    
    if isinstance(column_name, str):
        # Iterate through each row in the original DataFrame
        for index, row in df.iterrows():
            # Get the list from the specified column
            list_values = row[column_name]
            
            # Check if the value is actually a list
            if isinstance(list_values, list):
                # For each value in the list, create a new row
                for value in list_values:
                    # Make a copy of the original row
                    new_row = row.copy()
                    # Replace the list with the single value
                    new_row[column_name] = value
                    # Add to our collection of new rows
                    new_rows.append(new_row)
            else:
                # If the value is not a list, keep the row as is
                new_rows.append(row)
    elif isinstance(column_name, list):
        # Iterate through each row in the original DataFrame
        for index, row in df.iterrows():
            # Get the list from the specified column
            list_values = row[column_name[0]]
            
            # Check if the value is actually a list
            if isinstance(list_values, list):
                # For each value in the list, create a new row
                for i, value in enumerate(list_values):
                    # Make a copy of the original row
                    new_row = row.copy()
                    # Replace the list with the single value
                    for j, col in enumerate(column_name):
                        new_row[col] = row[column_name[j]][i]
                        
                    # Add to our collection of new rows
                    new_rows.append(new_row)
            else:
                # If the value is not a list, keep the row as is
                new_rows.append(row)
    
    # Create a new DataFrame from the collected rows
    result_df = pd.DataFrame(new_rows)
    
    return result_df.reset_index(drop=True)