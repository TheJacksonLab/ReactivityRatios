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

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
import seaborn as sns

from cheminfo import FMO_trans, gen_feature_df_paral
from cheminfo import judge_type, unpack_list_column
from utils.model import train_xgb_model_regressor

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


    result_df_split = filtered_result_df[['Monomer_1', 'Monomer_2', 'type'] + list(filtered_result_df.columns[2:-3])]

    indices = np.arange(filtered_result_df.shape[0])

    train_indices, test_indices = train_test_split(indices, test_size=test_size, 
                                                random_state=random_state, stratify=type_labels)

    train_label = unpack_list_column(filtered_result_df.iloc[train_indices,:], column_name='type')['type'].values
    test_label = unpack_list_column(filtered_result_df.iloc[test_indices,:], column_name='type')['type'].values

    # Unpack and prepare training data
    X_train = np.concatenate([
        unpack_list_column(result_df_split.iloc[train_indices,:], column_name='type').iloc[:, 4:].values
    ], axis=0)

    y_train = np.concatenate([
        unpack_list_column(result_df_split.iloc[train_indices,:], column_name='type')[['type']].values
    ], axis=0)
    y_train = np.eye(5)[y_train.flatten()]

    X_test = np.concatenate([
        unpack_list_column(result_df_split.iloc[test_indices,:], column_name=['type']).iloc[:, 4:].values], axis=0)

    y_test = np.concatenate([
        unpack_list_column(result_df_split.iloc[test_indices,:], column_name=['type'])[['type']].values
    ], axis=0)
    y_test = np.eye(5)[y_test.flatten()]

    # Preprocessing x and y
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = X_train_scaled
    X_test = X_test_scaled


    # Setup XGBoost model
    best_params = {
        'colsample_bytree': 0.9,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'n_estimators': 200,
        'reg_alpha': 0,
        'reg_lambda': 2,
        'subsample': 0.9
    }

    # Check if y_train is one-hot encoded
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        # Convert from one-hot encoding to class indices
        y_train = np.argmax(y_train, axis=1)
        if 'y_test' in locals() or 'y_test' in globals():
            y_test = np.argmax(y_test, axis=1)

    # Count unique classes
    num_classes = len(np.unique(y_train))


    best_xgb_model = XGBClassifier(
        random_state=random_state,
        objective='multi:softprob',  # For multi-class classification
        num_class=num_classes,       # Specify number of classes
        eval_metric='mlogloss',      # Multi-class log loss
        n_jobs=-1,
        **best_params
    )

    # Fit the model
    best_xgb_model.fit(X_train, y_train)

    y_train_pred = best_xgb_model.predict(X_train)
    y_test_pred = best_xgb_model.predict(X_test)
    
    training_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')


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

    type_labels = np.array(list(map(judge_type, test_result_df[['r1', 'r2']].values.tolist())))
    test_result_df['type'] = type_labels

    # type_labels = list(map(_get_mode, 
    #                 test_result_df['type'].values.tolist()))


    result_df_split = test_result_df[['Monomer_1', 'Monomer_2', 'type'] + list(test_result_df.columns[2:-3])]


    # Unpack and prepare training data
    X_test_cka = np.concatenate([
        unpack_list_column(result_df_split.iloc[:,:], column_name='type').iloc[:, 4:].values
    ], axis=0)

    y_test_cka = np.concatenate([
        unpack_list_column(result_df_split.iloc[:,:], column_name='type')[['type']].values
    ], axis=0)
    y_test_cka = y_test_cka.flatten()

    X_test_cka_scaled = scaler.transform(X_test_cka)
    X_test_cka = X_test_cka_scaled

    y_test_cka_pred = best_xgb_model.predict(X_test_cka)


    test_cka_f1 = f1_score(y_test_cka, y_test_cka_pred, average='weighted')
    test_cka_accuracy = accuracy_score(y_test_cka, y_test_cka_pred)


    result_summary = {
        "training_accuracy": training_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1": train_f1,
        "test_f1": test_f1,

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