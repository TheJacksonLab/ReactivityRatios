import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.smarts_manipulation import get_smi_prod
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from math import sqrt


HARTREE2KJ = 2625.5
R = 8.314


def search_inchikey(inchikey, worksheet, type_='int'):
    assert type_ in ['int', 'sm', 'ts']
    if type_=='int':
        prefix = ''
    elif type_=='sm':
        prefix = 'PreSm_'
    elif type_=='ts':
        prefix = 'PreTS_'
    assert "Name" in worksheet.columns, f"Name not in the worksheet"
    
    inchikey = prefix + inchikey
    
    if inchikey not in worksheet['Name'].values:
        return []
    
    res_inckikey = []

    if inchikey in worksheet['Name'].values:
        res_inckikey.append(inchikey)
        
    return res_inckikey

def get_discriptor(worksheet, inchikey, feature_list=['corr[au]','Hcorr[au]','Ucorr[au]','ZPE[au]','TS[au]']):
    assert "Name" in worksheet.columns, f"Name not in the worksheet"
    if inchikey not in worksheet['Name'].values:
        return None
    
    # Filter worksheet to get just the target molecule
    molecule_data = worksheet[worksheet['Name'] == inchikey]
    
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

def process_row(idx, exp_worksheet, mol_descriptors_rev, feature_list, capping_group, expected_mol_num):
    series = exp_worksheet.iloc[idx]
    mol1_name = series['Monomer_1']
    mol2_name = series['Monomer_2']
    r1 = series['r1']
    r2 = series['r2']
    T = series['Temperature (K)']
    
    mol1s = series['Monomer_1_smi']
    mol2s = series['Monomer_2_smi']
    rad1s = get_smi_prod(series['active_smi1'], capping_frag=capping_group)
    rad2s = get_smi_prod(series['active_smi2'], capping_frag=capping_group)
    ts11s, prod11s = series['smarts_11'].split('>>')
    ts12s, prod12s = series['smarts_12'].split('>>')
    ts21s, prod21s = series['smarts_21'].split('>>')
    ts22s, prod22s = series['smarts_22'].split('>>')
    ts11s = get_smi_prod(ts11s, capping_frag=capping_group)
    ts12s = get_smi_prod(ts12s, capping_frag=capping_group)
    ts21s = get_smi_prod(ts21s, capping_frag=capping_group)
    ts22s = get_smi_prod(ts22s, capping_frag=capping_group)
    prod11s = get_smi_prod(prod11s, capping_frag=capping_group)
    prod12s = get_smi_prod(prod12s, capping_frag=capping_group)
    prod21s = get_smi_prod(prod21s, capping_frag=capping_group)
    prod22s = get_smi_prod(prod22s, capping_frag=capping_group)
    
    mol1_inchiks = search_smiles(mol1s, mol_descriptors_rev, 'int')
    mol2_inchiks = search_smiles(mol2s, mol_descriptors_rev, 'int')
    rad1_inchiks = search_smiles(rad1s, mol_descriptors_rev, 'int')
    rad2_inchiks = search_smiles(rad2s, mol_descriptors_rev, 'int')
    ts11_inchiks = search_smiles(ts11s, mol_descriptors_rev, 'ts')
    ts12_inchiks = search_smiles(ts12s, mol_descriptors_rev, 'ts')
    ts21_inchiks = search_smiles(ts21s, mol_descriptors_rev, 'ts')
    ts22_inchiks = search_smiles(ts22s, mol_descriptors_rev, 'ts')
    sm11_inchiks = search_smiles(ts11s, mol_descriptors_rev, 'sm')
    sm12_inchiks = search_smiles(ts12s, mol_descriptors_rev, 'sm')
    sm21_inchiks = search_smiles(ts21s, mol_descriptors_rev, 'sm')
    sm22_inchiks = search_smiles(ts22s, mol_descriptors_rev, 'sm')
    
    
    # Alphabetically reorder the monomer1 and monomer2 columns
    if mol1_name > mol2_name:
        mol1_name, mol2_name = mol2_name, mol1_name
        r1, r2 = r2, r1
        mol1s, mol2s = mol2s, mol1s
        rad1s, rad2s = rad2s, rad1s
        mol1_inchiks, mol2_inchiks = mol2_inchiks, mol1_inchiks
        rad1_inchiks, rad2_inchiks = rad2_inchiks, rad1_inchiks
        ts11s, ts12s = ts22s, ts21s
        ts21s, ts22s = ts12s, ts11s
        ts11_inchiks, ts12_inchiks = ts22_inchiks, ts21_inchiks
        ts21_inchiks, ts22_inchiks = ts12_inchiks, ts11_inchiks

        sm11_inchiks, sm12_inchiks = sm22_inchiks, sm21_inchiks
        sm21_inchiks, sm22_inchiks = sm12_inchiks, sm11_inchiks
        
    
    final_inchiks = mol1_inchiks + mol2_inchiks + rad1_inchiks + rad2_inchiks + \
        ts11_inchiks + ts12_inchiks + ts21_inchiks + ts22_inchiks + \
        sm11_inchiks + sm12_inchiks + sm21_inchiks + sm22_inchiks
    
    res_features = pd.Series([mol1_name, mol2_name, r1, r2, T],
                          index=['Monomer_1', 'Monomer_2', 'r1', 'r2', 'T'])
    
    try:
        assert len(final_inchiks) == expected_mol_num, \
            f"len of inchikeys not equal to {expected_mol_num}, {final_inchiks}"
        for (prefix, inchik) in zip(["mol1_", "mol2_", "rad1_", "rad2_", 
                                     "ts11_", "ts12_", "ts21_", "ts22_", 
                                     "sm11_", "sm12_", "sm21_", "sm22_"], final_inchiks):
            des_line = get_discriptor(mol_descriptors_rev, inchik, feature_list=feature_list)
            # change the column name
            des_line.index = [prefix + i for i in des_line.index]
            
            # concat 2 series
            res_features = pd.concat([res_features, des_line], axis=0)
        
        return res_features
    except:
        return None
    
def gen_feature_df_paral(exp_worksheet, mol_descriptors_rev, feature_list, capping_group="[*]H", expected_mol_num=12):
    
    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    
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


    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_func, indices), total=len(indices)))
    
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
        final_df = pd.DataFrame(dataframe_value)
        return final_df
    else:
        return pd.DataFrame()
    
def compute_r1_r2(series, ZPE_corr=1.0):

    T = series['T']
    std_correction =  - 8.314 * T / 1000 * np.log(24.46)  # in kJ/mol
    mol1_G = (series['mol1_Hcorr[au]'] + series['mol1_SPE[au]'] - series['mol1_TS[au]'] / 298 * T - series['mol1_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    mol2_G = (series['mol2_Hcorr[au]'] + series['mol2_SPE[au]'] - series['mol2_TS[au]'] / 298 * T - series['mol2_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    rad1_G = (series['rad1_Hcorr[au]'] + series['rad1_SPE[au]'] - series['rad1_TS[au]'] / 298 * T - series['rad1_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    rad2_G = (series['rad2_Hcorr[au]'] + series['rad2_SPE[au]'] - series['rad2_TS[au]'] / 298 * T - series['rad2_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    ts11_G = (series['ts11_Hcorr[au]'] + series['ts11_SPE[au]'] - series['ts11_TS[au]'] / 298 * T - series['ts11_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    ts12_G = (series['ts12_Hcorr[au]'] + series['ts12_SPE[au]'] - series['ts12_TS[au]'] / 298 * T - series['ts12_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    ts21_G = (series['ts21_Hcorr[au]'] + series['ts21_SPE[au]'] - series['ts21_TS[au]'] / 298 * T - series['ts21_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    ts22_G = (series['ts22_Hcorr[au]'] + series['ts22_SPE[au]'] - series['ts22_TS[au]'] / 298 * T - series['ts22_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    sm11_G = (series['sm11_Hcorr[au]'] + series['sm11_SPE[au]'] - series['sm11_TS[au]'] / 298 * T - series['sm11_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    sm12_G = (series['sm12_Hcorr[au]'] + series['sm12_SPE[au]'] - series['sm12_TS[au]'] / 298 * T - series['sm12_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    sm21_G = (series['sm21_Hcorr[au]'] + series['sm21_SPE[au]'] - series['sm21_TS[au]'] / 298 * T - series['sm21_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    sm22_G = (series['sm22_Hcorr[au]'] + series['sm22_SPE[au]'] - series['sm22_TS[au]'] / 298 * T - series['sm22_ZPE[au]'] * (1. - ZPE_corr)) * HARTREE2KJ + std_correction
    
    G_act_11 = ts11_G - min(mol1_G + rad1_G, sm11_G)
    G_act_12 = ts12_G - min(mol2_G + rad1_G, sm12_G)
    G_act_21 = ts21_G - min(mol1_G + rad2_G, sm21_G)
    G_act_22 = ts22_G - min(mol2_G + rad2_G, sm22_G)
    
    ddG_1 = G_act_11 - G_act_12
    ddG_2 = G_act_22 - G_act_21
    # print(f"ddG_1: {ddG_1}, ddG_2: {ddG_2}")
    r1 = np.exp(-ddG_1 / ( R * T) * 1000) 
    r2 = np.exp(-ddG_2 / ( R * T) * 1000)
    return r1, r2

def plot_confusion_matrix(y_true, y_pred, fig_name="Confusion_Matrix_w.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm = np.nan_to_num(cm)
    cm = np.maximum(cm, 1E-6)  # Avoid division by zero
    cm = cm / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix
    # substitute NAN with 0
    cm = np.nan_to_num(cm)
    # padding cm to (5,5) with 0
    cm = np.pad(cm, ((0, 5 - cm.shape[0]), (0, 5 - cm.shape[1])), mode='constant', constant_values=0)

    plt.figure(figsize=(8, 6))
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True, center="light")  # Blue to orange
    type2label = {"alternating":0, "random":1, "gradient":2, "pseudo-alternating":3, "blocky":4 }
    # Create the heatmap with larger annotation fonts
    ax = sns.heatmap(
        cm, 
        annot=True, 
        cmap=cmap,  
        xticklabels=type2label.keys(), 
        yticklabels=type2label.keys(),
        annot_kws={"size": 14}  # Increase annotation font size
    )

    # Increase font size for x and y tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    
    # plt.xlabel('Predicted Type', fontsize=14)
    # plt.ylabel('Actual Type', fontsize=14)
    # plt.title('Normalized Confusion Matrix', fontdict={'fontsize': 16, 'fontname': 'Arial'})
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

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