import pandas as pd
import numpy as np
import os, sys
from glob import glob
import pickle
import rdkit
from rdkit import Chem

# Add the parent directory to the system path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("..")
# Convert relative imports to absolute imports
from utils.descriptor_list import dft_els_prop, dft_FMO, dft_surface, dft_size_shape
from utils.descriptor_list import rdkit_descriptors, reac_BV_descriptors
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
from utils.smarts_manipulation import get_smi_prod
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
import seaborn as sns

from cheminfo import FMO_trans, gen_feature_df_paral
from cheminfo import judge_type, unpack_list_column
from utils.model import train_dt_model_regressor

def main(random_state=42):
    mol_descriptors_path = "/Users/jingdan/Desktop/scrugs/project/0418_capping/DFT_cappingC.parquet"
    unrelated_descriptors = ["react_atom_serial", ]
    exp_worksheet = pd.read_csv('../data/monomer_react_ratio_smarts.csv')
    capping_group = "[*]C"
    capping_name = capping_group.replace("*", "").replace("[", "").replace("]", "")
    excepted_mol_num = 4
    exist_ignore = True
    r_low, r_high = 0., 50.
    test_size = 0.2

    feature_list = dft_FMO + dft_size_shape 
    assert len(feature_list) == len(set(feature_list)), "Duplicated features in feature_list"

    mol_descriptors = pd.read_parquet(mol_descriptors_path)
    mol_smiles = mol_descriptors['smiles']
    mol_descriptors_rev = mol_descriptors.apply(FMO_trans, axis=1)
    mol_descriptors_rev.drop(columns=unrelated_descriptors + [
        'HOMO_a', 'HOMO_b', 'LUMO_a', 'LUMO_b', 'ODI_HOMO_a', 'ODI_HOMO_b', 'ODI_LUMO_a', 'ODI_LUMO_b','smiles'
        ], inplace=True)

    # Create .cache directory if it does not exist
    os.makedirs(".cache", exist_ok=True)
    if exist_ignore or (not os.path.exists(f".cache/feature_df_capping{capping_name}.parquet")):
        # Generate the feature DataFrame
        result_df = gen_feature_df_paral(exp_worksheet, mol_descriptors_rev, feature_list, 
                                        capping_group=capping_group, expected_mol_num=excepted_mol_num)
        # Save the result to a parquet file
        result_df.to_parquet(f".cache/feature_df_capping{capping_name}.parquet")
    else:
        result_df = pd.read_parquet(f".cache/feature_df_capping{capping_name}.parquet")

    filtered_result_df = result_df.dropna(subset=['r1', 'r2'])

    filtered_result_df = filtered_result_df[
        (result_df['r1'] > r_low) & 
        (result_df['r1'] < r_high) & 
        (result_df['r2'] > r_low) & 
        (result_df['r2'] < r_high) ].reindex()

    r_pair_array = filtered_result_df[['r1', 'r2']].values
    type_labels = np.array(list(map(judge_type, r_pair_array.tolist())))
    filtered_result_df['type'] = type_labels

    df_feature_names = list(result_df.columns[4:])
    df_target_names = ['r1', 'r2', 'type']

    def _merge_dict(d1, d2):
        # Merge two dictionaries
        d1.update(d2)
        return d1

    def _get_mode(data_list):
        # Get the mode of a list
        if len(data_list) == 0:
            return None
        else:
            return max(set(data_list), key=data_list.count)

    filtered_result_df = filtered_result_df.groupby(['Monomer_1', 'Monomer_2']).agg(_merge_dict(
        {feature: "mean" for feature in df_feature_names},
        {target: lambda x: list(x) for target in df_target_names}
    )).reset_index()

    type_labels = list(map(_get_mode, 
                    filtered_result_df['type'].values.tolist()))

    # Create lists to store columns for each dataframe
    split1_data, split2_data = {}, {}

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
    np.random.seed(random_state)

    train_indices, test_indices = train_test_split(indices, test_size=test_size, 
                                                random_state=random_state, stratify=type_labels)

    train_label = unpack_list_column(filtered_result_df.iloc[train_indices,:], column_name='type')['type'].values
    test_label = unpack_list_column(filtered_result_df.iloc[test_indices,:], column_name='type')['type'].values

    # Unpack and prepare training data
    X_train = np.concatenate([
        unpack_list_column(result_df_split1.iloc[train_indices,:]).iloc[:, 2:].values,
        unpack_list_column(result_df_split2.iloc[train_indices,:]).iloc[:, 2:].values
    ], axis=0)

    y_train = np.concatenate([
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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = X_train_scaled
    X_test = X_test_scaled
    all_X = np.concatenate([X_train, X_test], axis=0)
    all_y = np.concatenate([y_train, y_test], axis=0)

    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    all_y_log = np.log(all_y)

    decision_tree_params = {
        'max_depth': 10,               # Same as your XGBoost setting
        'min_samples_split': 2,        # Similar to min_child_weight
        'min_samples_leaf': 1,         # Approximately equivalent to min_child_weight=1
        'max_features': 0.9,           # Similar to colsample_bytree=0.9
        'splitter': 'best',            # Default value
        'criterion': 'squared_error',           # For classification (use 'mse' for regression)
        'min_impurity_decrease': 0.1,  # Similar to gamma=0.1
        'ccp_alpha': 0.0               # Similar to reg_alpha=0
    }

    best_xgb_model = train_dt_model_regressor(
        X_train, y_train_log, 
        params=decision_tree_params,
        random_state=random_state,)

    y_train_log_pred = best_xgb_model.predict(X_train)
    y_test_log_pred  = best_xgb_model.predict(X_test)
    y_train_pred = np.exp(y_train_log_pred)
    y_test_pred  = np.exp(y_test_log_pred)

    train_label_predict = np.array(list(map(
        judge_type, np.concatenate(
            [y_train_pred[:y_train_pred.shape[0]//2][:,None], y_train_pred[y_train_pred.shape[0]//2:][:,None]],
            axis=1).tolist())))
    test_label_predict = np.array(list(map(
        judge_type, np.concatenate(
            [y_test_pred[:y_test_pred.shape[0]//2][:, None], y_test_pred[y_test_pred.shape[0]//2:][:, None]],
            axis=1).tolist())))

    train_log_mse = mean_squared_error(y_train_log, y_train_log_pred)
    test_log_mse  = mean_squared_error(y_test_log, y_test_log_pred)
    train_log_rmse = np.sqrt(train_log_mse)
    test_log_rmse  = np.sqrt(test_log_mse)
    train_log_mae = mean_absolute_error(y_train_log, y_train_log_pred)
    test_log_mae  = mean_absolute_error(y_test_log, y_test_log_pred)
    train_log_r2 = r2_score(y_train_log, y_train_log_pred)
    test_log_r2  = r2_score(y_test_log, y_test_log_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse  = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse  = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae  = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2  = r2_score(y_test, y_test_pred)

    training_accuracy = np.sum(train_label_predict == train_label) / len(train_label)
    test_accuracy = np.sum(test_label_predict == test_label) / len(test_label)
    train_f1 = f1_score(train_label, train_label_predict, average='weighted')
    test_f1 = f1_score(test_label, test_label_predict, average='weighted')


    test_mol_descriptors_path = "/Users/jingdan/Desktop/scrugs/project/0418_capping/DFT_CKAs_cappingC.parquet"
    unrelated_descriptors = ["react_atom_serial", ]
    test_exp_worksheet = pd.read_csv('../data/CKAs.csv')

    capping_group = "[*]C"
    excepted_mol_num = 4

    test_mol_descriptors = pd.read_parquet(test_mol_descriptors_path)

    # Apply the FMO_trans function to each row of the dataframe
    test_mol_descriptors_rev = test_mol_descriptors.apply(FMO_trans, axis=1)
    test_mol_descriptors_rev.drop(columns=unrelated_descriptors + [
        'HOMO_a', 'HOMO_b', 'LUMO_a', 'LUMO_b', 'ODI_HOMO_a', 'ODI_HOMO_b', 'ODI_LUMO_a', 'ODI_LUMO_b','smiles'
        ], inplace=True)

    if exist_ignore or (not os.path.exists(f".cache/CKAs_feature_df_capping{capping_name}.parquet")):
        # Generate the feature DataFrame
        test_result_df = gen_feature_df_paral(test_exp_worksheet, test_mol_descriptors_rev, feature_list, 
                                            capping_group=capping_group, expected_mol_num=excepted_mol_num)
        # Save the result to a parquet file
        test_result_df.to_parquet(f".cache/CKAs_feature_df_capping{capping_name}.parquet")
    else:
        test_result_df = pd.read_parquet(f".cache/CKAs_feature_df_capping{capping_name}.parquet")

    # Create lists to store columns for each dataframe
    split1_data = {}
    split2_data = {}

    # Add monomer names and r values
    monomer_names = test_result_df["Monomer_1"] + test_result_df["Monomer_2"]
    split1_data["Monomer"] = monomer_names
    split2_data["Monomer"] = monomer_names
    split1_data["r"] = test_result_df["r1"]
    split2_data["r"] = test_result_df["r2"]

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
                    split1_data[feature] = test_result_df[feature]
                
                # Add to split2 with appropriate prefix replacement
                if mapping["split2"]:
                    target_prefix, source_prefix = mapping["split2"]
                    target_feature = feature.replace(prefix, target_prefix)
                    split2_data[target_feature] = test_result_df[feature]
                
                break  # Move to next feature after finding a match

    # Create DataFrames all at once from the collected data
    result_df_split1 = pd.DataFrame(split1_data)
    result_df_split2 = pd.DataFrame(split2_data)

    # re-sort the columns of reasult_df_split2 as result_df_split1
    result_df_split2 = result_df_split2[result_df_split1.columns]

    X_test_cka = np.concatenate([
        unpack_list_column(result_df_split1).iloc[:, 2:].values,
        unpack_list_column(result_df_split2).iloc[:, 2:].values
    ], axis=0)

    y_test_cka = np.concatenate([
        unpack_list_column(result_df_split1)['r'].values,
        unpack_list_column(result_df_split2)['r'].values
    ], axis=0)

    X_test_cka_scaled = scaler.transform(X_test_cka)
    y_test_cka_log_pred = best_xgb_model.predict(X_test_cka_scaled)

    y_test_cka_pred = np.exp(y_test_cka_log_pred)
    y_test_cka_log = np.log(y_test_cka)

    test_cka_label_predict = np.array(list(map(
        judge_type, np.concatenate(
            [y_test_cka_pred[:y_test_cka_pred.shape[0]//2][:, None], y_test_cka_pred[y_test_cka_pred.shape[0]//2:][:, None]],
            axis=1).tolist())))
    test_cka_label = np.array(list(map(
        judge_type, np.concatenate(
            [y_test_cka[:y_test_cka.shape[0]//2][:, None], y_test_cka[y_test_cka.shape[0]//2:][:, None]],
            axis=1).tolist())))

    # compute evaluation metrics
    test_cka_log_mse = mean_squared_error(y_test_cka_log, y_test_cka_log_pred)
    test_cka_log_rmse = np.sqrt(test_cka_log_mse)
    test_cka_log_mae = mean_absolute_error(y_test_cka_log, y_test_cka_log_pred)
    test_cka_log_r2 = r2_score(y_test_cka_log, y_test_cka_log_pred)

    test_cka_r2 = r2_score(y_test_cka, y_test_cka_pred)
    test_cka_mse = mean_squared_error(y_test_cka, y_test_cka_pred)
    test_cka_rmse = np.sqrt(test_cka_mse)
    test_cka_mae = mean_absolute_error(y_test_cka, y_test_cka_pred)

    test_cka_f1 = f1_score(test_cka_label, test_cka_label_predict, average='weighted')
    test_cka_accuracy = np.sum(test_cka_label_predict == test_cka_label) / len(test_cka_label)

    result_summary = {
        "train_log_rmse": train_log_rmse,
        "test_log_rmse": test_log_rmse,
        "train_log_mae": train_log_mae,
        "test_log_mae": test_log_mae,
        "train_log_r2": train_log_r2,
        "test_log_r2": test_log_r2,

        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,

        "training_accuracy": training_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1": train_f1,
        "test_f1": test_f1,

        'test_cka_log_rmse': test_cka_log_rmse,
        'test_cka_log_mae': test_cka_log_mae,
        'test_cka_log_r2': test_cka_log_r2,

        'test_cka_rmse': test_cka_rmse,
        'test_cka_mae': test_cka_mae,
        'test_cka_r2': test_cka_r2,

        'test_cka_f1': test_cka_f1,
        'test_cka_accuracy': test_cka_accuracy
    }
    return result_summary

if __name__ == "__main__":
    random_state_range = list(range(40, 50))
    all_results = []
    for random_state in random_state_range:
        # print(f"random_state: {random_state}")
        result_summary = main(random_state=random_state)
        all_results.append(result_summary)
        # break

    all_results_df = pd.DataFrame(all_results)

    # compute the mean and std of each column
    mean_results = all_results_df.mean()
    std_results = all_results_df.std()
    
    result_text = ""
    for col in all_results_df.columns:
        result_text += f"{col}: {mean_results[col]:.4f} ± {std_results[col]:.4f}\n"

    out_filename = os.path.basename(__file__)
    out_filename = out_filename.replace(".py", ".results.txt")
    with open(out_filename, "w") as f:
        f.write(result_text)