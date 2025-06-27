import pandas as pd
import numpy as np
import os, sys
from glob import glob
import pickle
import rdkit
from rdkit import Chem
from descriptor_list import dft_els_prop, dft_FMO, dft_surface, dft_size_shape
from descriptor_list import rdkit_descriptors, reac_BV_descriptors

# import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
from functools import partial
from utils.smarts_manipulation import get_smi_prod
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score
import json

import warnings
warnings.filterwarnings("ignore")

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)

# for a hash fix
from glob import glob

json_files = glob("ChemH_saved_model/xgb_scores_*.json")

def read_json(filename):
    with open(filename, "r") as f:
        cont = json.load(f)

    info = cont["info"]
    return (info['capping_group'], info['feature_comb_key'], info['r_low'], info['r_high'], info['test_size']), info['hash_value']

cache_dict = {}

for json_file in json_files:
    param_set, hash_value = read_json(json_file)
    capping_group = param_set[0]
    capping_name = capping_group.replace("*", "").replace("[", "").replace("]", "")
    cache_dict[param_set] = f".cache/feature_df_capping{capping_name}_{hash_value}.parquet"

# Selection pool
capping_groups = ["[*]C", "[*]OC", "[*][H]"]
r_lows = [0.]
r_highs = [50.]
test_sizes = [0.2]
# Create dictionary with descriptive names as keys
feature_dict = {
    "rdkit_only": rdkit_descriptors,
    "dft_els_prop": dft_els_prop,
    "dft_FMO": dft_FMO,
    "dft_surface": dft_surface,
    "dft_size_shape": dft_size_shape,
    "dft_els_prop_FMO": dft_els_prop + dft_FMO,
    "dft_els_prop_surface": dft_els_prop + dft_surface,
    "dft_els_prop_size_shape": dft_els_prop + dft_size_shape,
    "dft_FMO_surface": dft_FMO + dft_surface,
    "dft_FMO_size_shape": dft_FMO + dft_size_shape,
    "dft_surface_size_shape": dft_surface + dft_size_shape,
    "dft_els_prop_FMO_surface": dft_els_prop + dft_FMO + dft_surface,
    "dft_els_prop_FMO_size_shape": dft_els_prop + dft_FMO + dft_size_shape,
    "dft_els_prop_surface_size_shape": dft_els_prop + dft_surface + dft_size_shape,
    "dft_FMO_surface_size_shape": dft_FMO + dft_surface + dft_size_shape,
    "dft_all": dft_els_prop + dft_FMO + dft_surface + dft_size_shape,
    "dft_all_reac_BV": dft_els_prop + dft_FMO + dft_surface + dft_size_shape + reac_BV_descriptors,
    "dft_all_rdkit": dft_els_prop + dft_FMO + dft_surface + dft_size_shape + rdkit_descriptors,
    "dft_FMO_reac_BV": dft_FMO + reac_BV_descriptors,
    "dft_FMO_rdkit": dft_FMO + rdkit_descriptors,
    "dft_all_reac_BV": dft_els_prop + dft_FMO + dft_surface + dft_size_shape + reac_BV_descriptors,
    "dft_FMO_reac_BV": dft_FMO + reac_BV_descriptors,
} # in-total 22x combinations

# xgb_param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [6, 8, 10],
#     'subsample': [0.7, 0.9],
#     "learning_rate": [0.05, 0.1],
#     "min_child_weight": [1, 3],
#     "gamma": [0, 0.1],
#     'reg_alpha': [0, 0.1],
#     'reg_lambda': [1, 2],
#     'colsample_bytree': [0.8, 0.9],
#     'huber_slope': [1., .8, .6],
# }

# Unchanged Params
exp_worksheet = pd.read_csv('data/monomer_react_ratio_smarts.csv')
excepted_mol_num = 4
exist_ignore = False


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

def gen_hash(feature_list, capping_group, expected_mol_num, r_min, r_max):
    # hash all input into a identifier
    hash_tuple = tuple(feature_list + [capping_group, expected_mol_num, r_min, r_max])
    return hash(hash_tuple)
    
def process_row(idx, exp_worksheet, mol_descriptors_rev, feature_list, capping_group, expected_mol_num):
    series = exp_worksheet.iloc[idx]
    mol1_name = series['Monomer_1']
    mol2_name = series['Monomer_2']
    r1 = series['r1']
    r2 = series['r2']
    
    mol1s = series['Monomer_1_smi']
    mol2s = series['Monomer_2_smi']
    rad1s = get_smi_prod(series['active_smi1'], capping_frag=capping_group)
    rad2s = get_smi_prod(series['active_smi2'], capping_frag=capping_group)
    
    mol1_inchiks = search_smiles(mol1s, mol_descriptors_rev, 'int')
    mol2_inchiks = search_smiles(mol2s, mol_descriptors_rev, 'int')
    rad1_inchikss = search_smiles(rad1s, mol_descriptors_rev, 'int')
    rad2_inchiks = search_smiles(rad2s, mol_descriptors_rev, 'int')
    
    # Alphabetically reorder the monomer1 and monomer2 columns
    if mol1_name > mol2_name:
        mol1_name, mol2_name = mol2_name, mol1_name
        r1, r2 = r2, r1
        mol1s, mol2s = mol2s, mol1s
        rad1s, rad2s = rad2s, rad1s
        mol1_inchiks, mol2_inchiks = mol2_inchiks, mol1_inchiks
        rad1_inchikss, rad2_inchiks = rad2_inchiks, rad1_inchikss
    
    final_inchiks = mol1_inchiks + mol2_inchiks + rad1_inchikss + rad2_inchiks
    
    res_features = pd.Series([mol1_name, mol2_name, r1, r2],
                          index=['Monomer_1', 'Monomer_2', 'r1', 'r2'])
    
    try:
        assert len(final_inchiks) == expected_mol_num, \
            f"len of inchikeys not equal to {expected_mol_num}, {final_inchiks}"
        for (prefix, inchik) in zip(["mol1_", "mol2_", "rad1_", "rad2_"], final_inchiks):
            des_line = get_discriptor(mol_descriptors_rev, inchik, feature_list=feature_list)
            # change the column name
            des_line.index = [prefix + i for i in des_line.index]
            
            # concat 2 series
            res_features = pd.concat([res_features, des_line], axis=0)
        
        return res_features
    except:
        return None

def gen_feature_df_paral(exp_worksheet, mol_descriptors_rev, feature_list, capping_group="[*]C", expected_mol_num=4, show_progress=True):
    
    # Assuming these variables are defined elsewhere or should be passed as arguments
    # exp_worksheet, mol_descriptors_rev, feature_list
    
    # Get the number of CPU cores
    # num_cores = multiprocessing.cpu_count()
    num_cores = 1
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_row, 
        exp_worksheet=exp_worksheet, 
        mol_descriptors_rev=mol_descriptors_rev, 
        feature_list=feature_list,
        capping_group=capping_group,
        expected_mol_num=expected_mol_num
    )
    
    # List to store results
    dataframe_value = []
    error_count = 0
    success_count = 0
    
    # Create a list of indices to process
    indices = list(range(exp_worksheet.shape[0]))
    
    # Use ProcessPoolExecutor for parallelization
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map the process_func to the indices
        if show_progress:
            results = list(tqdm(executor.map(process_func, indices), total=len(indices)))
        else:
            results = list(executor.map(process_func, indices))
    
    # Process results
    for result in results:
        if result is not None:
            dataframe_value.append(result)
            success_count += 1
        else:
            error_count += 1
    
    # print(f"num of success: {success_count}")
    # print(f"num of error: {error_count}")
    
    # Combine all results into a single DataFrame if needed
    if dataframe_value:
        # final_df = pd.concat(dataframe_value, axis=1, keys=range(len(dataframe_value))).T
        final_df = pd.DataFrame(dataframe_value)
        return final_df
    else:
        return pd.DataFrame()

def FMO_trans(series):
    # descriptor trans, designed for different descriptor calculation methods or FMO information for unrestricted/ restricted DFT
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
    if inchikey not in worksheet['molecule_inchikey_id'].values:
        return []
    
    res_inckikey = []

    if inchikey in worksheet['molecule_inchikey_id'].values:
        res_inckikey.append(inchikey)
    
    inchikey1 = 'p1_' + inchikey
    
    if inchikey1 in worksheet['molecule_inchikey_id'].values:
        res_inckikey.append(inchikey1)
        
    inchikey2 = 'n1_' + inchikey
    
    if inchikey2 in worksheet['molecule_inchikey_id'].values:
        res_inckikey.append(inchikey2)
        
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

def _get_mode(data_list):
    # Get the mode of a list
    if len(data_list) == 0:
        return None
    else:
        return max(set(data_list), key=data_list.count)

def unpack_list_column(df, column_name='r'):
    """
    Unpacks a DataFrame column containing lists into multiple rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the list column to unpack
    column_name : str, default='r'
        The name of the column containing lists
        
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the list column unpacked
    """
    # Create a list to store the new rows
    new_rows = []
    
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
    
    # Create a new DataFrame from the collected rows
    result_df = pd.DataFrame(new_rows)
    
    return result_df.reset_index(drop=True)

def _merge_dict(d1, d2):
    # Merge two dictionaries
    d1.update(d2)
    return d1


def main(capping_group, feature_list, r_low, r_high, test_size):
    
    capping_name = capping_group.replace("*", "").replace("[", "").replace("]", "")
    mol_descriptors_path = f"/Users/jingdan/Desktop/scrugs/project/0418_capping/DFT_capping{capping_name}.parquet"

    assert len(feature_list) == len(set(feature_list)), "Duplicated features in feature_list"
    mol_descriptors = pd.read_parquet(mol_descriptors_path)
    mol_smiles = mol_descriptors['smiles']

    # Apply the FMO_trans function to each row of the dataframe
    unrelated_descriptors = ["react_atom_serial", ]
    mol_descriptors_rev = mol_descriptors.apply(FMO_trans, axis=1)
    mol_descriptors_rev.drop(columns=unrelated_descriptors + [
        'HOMO_a', 'HOMO_b', 'LUMO_a', 'LUMO_b', 'ODI_HOMO_a', 'ODI_HOMO_b', 'ODI_LUMO_a', 'ODI_LUMO_b','smiles'
        ], inplace=True)

    hash_value = gen_hash(feature_list, capping_group, excepted_mol_num, r_low, r_high)

    # (info['capping_group'], info['feature_comb_key'], info['r_low'], info['r_high'], info['test_size']), info['hash_value']
    param_set_key = (capping_group, feature_comb_key, r_low, r_high, test_size)
    
    if exist_ignore or (not os.path.exists(f".cache/feature_df_capping{capping_name}_{hash_value}.parquet")):
        # Generate the feature DataFrame
        result_df = gen_feature_df_paral(exp_worksheet, mol_descriptors_rev, feature_list, 
                                        capping_group=capping_group, expected_mol_num=excepted_mol_num, show_progress=True)
        
        # Save the result to a parquet file
        result_df.to_parquet(f".cache/feature_df_capping{capping_name}_{hash_value}.parquet")
    elif os.path.exists(cache_dict[param_set_key]):
        result_df = pd.read_parquet(cache_dict[param_set_key])
    else:
        result_df = pd.read_parquet(f".cache/feature_df_capping{capping_name}_{hash_value}.parquet")
        
        
    df_feature_names = list(result_df.columns[4:])
    df_target_names = ['r1', 'r2', 'type']

    # Dataseries filtering 
    # Filter 1: Filter out rows where 'r1' or 'r2' is NaN, kept r1, 12 between 0.02 to 20.
    filtered_result_df = result_df.dropna(subset=['r1', 'r2'])

    filtered_result_df = filtered_result_df[
        (result_df['r1'] > r_low) & (result_df['r1'] < r_high) & (result_df['r2'] > r_low) & (result_df['r2'] < r_high) ].reindex()

    r_pair_array = filtered_result_df[['r1', 'r2']].values
    type_labels = np.array(list(map(judge_type, r_pair_array.tolist())))
    filtered_result_df['type'] = type_labels

    # # Filter 2: merge same monomer pair with average r1, r2
    # filtered_result_df = filtered_result_df.groupby(['Monomer_1', 'Monomer_2']).agg({
    #     feature: "mean" for feature in df_feature_names + df_target_names}).reset_index()
    filtered_result_df = filtered_result_df.groupby(['Monomer_1', 'Monomer_2']).agg(_merge_dict(
        {feature: "mean" for feature in df_feature_names},
        {target: lambda x: list(x) for target in df_target_names}
    )).reset_index()
    


    type_labels = list(map(_get_mode, filtered_result_df['type'].values.tolist()))

    # Create lists to store columns for each dataframe
    split1_data = {}
    split2_data = {}

    # Add monomer names and r values
    monomer_names = filtered_result_df["Monomer_1"] + filtered_result_df["Monomer_2"]
    split1_data["Monomer"] = monomer_names
    split2_data["Monomer"] = monomer_names
    split1_data["r"] = filtered_result_df["r1"]
    split2_data["r"] = filtered_result_df["r2"]
    


    # Create prefix mappings for cleaner column handling
    prefix_mappings = {
        "mol1_": {"split1": True, "split2": ("mol2_", "mol1_")},
        "mol2_": {"split1": True, "split2": ("mol1_", "mol2_")},
        "rad1_": {"split1": True, "split2": None},
        "rad2_": {"split1": None, "split2": ("rad1_", "rad2_")}
    }

    # Process features based on prefix mappings
    for feature in df_feature_names:
        for prefix, mapping in prefix_mappings.items():
            if feature.startswith(prefix):
                # Add to split1 if mapping exists
                if mapping["split1"]:
                    split1_data[feature] = filtered_result_df[feature]
                
                # Add to split2 with appropriate prefix replacement
                if mapping["split2"]:
                    target_prefix, source_prefix = mapping["split2"]
                    target_feature = feature.replace(prefix, target_prefix)
                    split2_data[target_feature] = filtered_result_df[feature]
                
                break  # Move to next feature after finding a match

    # Create DataFrames all at once from the collected data
    result_df_split1 = pd.DataFrame(split1_data)
    result_df_split2 = pd.DataFrame(split2_data)

    # re-sort the columns of reasult_df_split2 as result_df_split1
    result_df_split2 = result_df_split2[result_df_split1.columns]

    indices = np.arange(filtered_result_df.shape[0])

    train_indices, test_indices = train_test_split(indices, test_size=test_size, 
                                                random_state=42, stratify=type_labels)

    train_label = unpack_list_column(filtered_result_df.iloc[train_indices,:], column_name='type')['type'].values
    test_label = unpack_list_column(filtered_result_df.iloc[test_indices,:], column_name='type')['type'].values
    
    X_train_outer = np.concatenate([
        unpack_list_column(result_df_split1.iloc[train_indices,:]).iloc[:, 2:].values,
        unpack_list_column(result_df_split2.iloc[train_indices,:]).iloc[:, 2:].values
    ], axis=0)

    y_train_outer = np.concatenate([
        unpack_list_column(result_df_split1.iloc[train_indices,:])['r'].values,
        unpack_list_column(result_df_split2.iloc[train_indices,:])['r'].values
    ], axis=0)

    X_test = np.concatenate([
        unpack_list_column(result_df_split1.iloc[test_indices,:]).iloc[:, 2:].values,
        unpack_list_column(result_df_split2.iloc[test_indices,:]).iloc[:, 2:].values
    ], axis=0)

    y_test = np.concatenate([
        unpack_list_column(result_df_split1.iloc[test_indices,:])['r'].values,
        unpack_list_column(result_df_split2.iloc[test_indices,:])['r'].values
    ], axis=0)
    
    # Preprocessing x and y
    scaler = StandardScaler()
    X_train_outer_scaled = scaler.fit_transform(X_train_outer)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_outer_log = np.log(y_train_outer)
    y_test_log = np.log(y_test)
    
    best_params = {
        'learning_rate': 0.05,
        'n_estimators': 300,
        'max_depth': 10,
        'min_child_weight': 1,
        'gamma': 0.,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
    }
    
    # Train final model with best parameters on all training data
    best_xgb_model = XGBRegressor(
        **best_params,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1,
        eval_metric='rmse'
    )
    
    best_xgb_model.fit(X_train_outer_scaled, y_train_outer_log)
    
    # Make predictions
    y_train_log_pred = best_xgb_model.predict(X_train_outer_scaled)
    y_test_log_pred = best_xgb_model.predict(X_test_scaled)
    y_train_pred = np.exp(y_train_log_pred)
    y_test_pred = np.exp(y_test_log_pred)
    
    # Compute reaction type labels from predictions
    train_label_predict = np.array(list(map(
        judge_type, np.concatenate(
            [y_train_pred[:y_train_pred.shape[0]//2][:,None], y_train_pred[y_train_pred.shape[0]//2:][:,None]],
            axis=1).tolist())))
    test_label_predict = np.array(list(map(
        judge_type, np.concatenate(
            [y_test_pred[:y_test_pred.shape[0]//2][:, None], y_test_pred[y_test_pred.shape[0]//2:][:, None]],
            axis=1).tolist())))

    # Compute evaluation metrics
    train_log_mse = mean_squared_error(y_train_outer_log, y_train_log_pred)
    test_log_mse = mean_squared_error(y_test_log, y_test_log_pred)

    train_log_rmse = np.sqrt(train_log_mse)
    test_log_rmse = np.sqrt(test_log_mse)

    train_log_mae = mean_absolute_error(y_train_outer_log, y_train_log_pred)
    test_log_mae = mean_absolute_error(y_test_log, y_test_log_pred)

    train_log_r2 = r2_score(y_train_outer_log, y_train_log_pred)
    test_log_r2 = r2_score(y_test_log, y_test_log_pred)

    # Compute evaluation metrics (exponential scale)
    train_mse = mean_squared_error(y_train_outer, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train_outer, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train_outer, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    training_accuracy = np.sum(train_label_predict == train_label) / len(train_label)
    test_accuracy = np.sum(test_label_predict == test_label) / len(test_label)

    f1 = f1_score(test_label, test_label_predict, average='weighted')

    # Compile scores into a dictionary
    scores = {
        'train_log_rmse': train_log_rmse,
        'test_log_rmse': test_log_rmse,
        'train_log_mae': train_log_mae,
        'test_log_mae': test_log_mae,
        'train_log_r2': train_log_r2,
        'test_log_r2': test_log_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'training_accuracy': training_accuracy,
        'test_accuracy': test_accuracy,
        'f1_score': f1,
    }
    
    return best_xgb_model, scores
if __name__ == "__main__":
    
    # Example usage
    for capping_group in capping_groups: # 3 choices
        for feature_comb_key in feature_dict: # 20 choices
            for r_low in r_lows: # 1 choices
                for r_high in r_highs: # 1 choices
                    for test_size in test_sizes: # 1 choices
                        from time import time
                        t0 = time()

                        feature_list = feature_dict[feature_comb_key]
                        hash_tup = tuple([capping_group, r_low, r_high, test_size, feature_comb_key])
                        hash_value = abs(hash(hash_tup))
                        
                        info_dict = {
                            "capping_group": capping_group,
                            "feature_comb_key": feature_comb_key,
                            "r_low": r_low,
                            "r_high": r_high,
                            "test_size": test_size,
                            "hash_value": hash_value
                        }
                        
                        print(f"Running with capping_group={capping_group}, feature_comb={feature_comb_key}, "
                              f"r_low={r_low}, r_high={r_high}, test_size={test_size}, hash_value={hash_value}")
                        
                        json_path = f"ChemH_saved_model_ChemC/xgb_scores_{hash_value}.json"
                        if os.path.exists(json_path):
                            print(f"File {json_path} already exists. Skipping this combination.")
                            continue

                        model, scores = main(capping_group, feature_list, r_low, r_high, test_size)
                        scores['info'] = info_dict
                        
                        # Save the model and scores
                        model_path = f"ChemH_saved_model_ChemC/xgb_model_{hash_value}.pkl"
                        pickle.dump(model, open(model_path, "wb"))
                        
                        # save scores to a file
                        
                        with open(json_path, 'w') as f:
                            json.dump(scores, f)
                        print(f"Model and scores saved for hash_value={hash_value}")

                        t1 = time()
                        print(f"Time taken: {t1 - t0:.2f} seconds")