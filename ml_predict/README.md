# Machine Learning Prediction

Machine learning models for predicting copolymerization reactivity ratios using molecular descriptors.

## Overview

Implements multiple ML algorithms to predict reactivity ratios from DFT-calculated molecular descriptors. Supports both regression (r1, r2 values) and classification (copolymerization type).

## Key Scripts

### Core Models
- **`xgboost_MR_task3.py`** - Main XGBoost regression model for reactivity ratio prediction
- **`ANN_MR_task3.py`** - Neural network models with PyTorch
- **`SVM_MR_task3.py`** - Support vector machine regression
- **`decisonT_MR_task3.py`** - Decision tree models

### Model Variants  
- **`xgboost_MR_task1.py`** - Individual reactivity ratio prediction
- **`xgboost_MR_task2.py`** - Alternative feature sets
- **`xgboost_MR_task3_wb97xd.py`** - High-level theory descriptors
- **`xgboost_MR_cv_task3.py`** - Cross-validation analysis

### Specialized Applications
- **`monomer_extrap_ChemC_cappingC_final.py`** - Chemical extrapolation to novel monomers
- **`chem_hyperp_exp_ChemC.py`** - Chemical space hyperparameter exploration
- **`cheminfo.py`** - Chemical informatics utilities for feature engineering

## Dependencies

```bash
pip install xgboost scikit-learn torch pandas numpy matplotlib seaborn shap
```

## Basic Usage

```python
# Train XGBoost model
python xgboost_MR_task3.py

# Train neural network  
python ANN_MR_task3.py

# Chemical extrapolation study
python monomer_extrap_ChemC_cappingC_final.py
```

## Model Performance

The main XGBoost model (`xgboost_MR_task3.py`) achieves:
- **R² > 0.85** for reactivity ratio prediction
- **85%+ accuracy** for copolymerization type classification
- Validated on external CKA dataset

## Key Features

### Data Processing
- Automatic feature engineering from molecular descriptors
- Data augmentation through symmetrical splitting
- Standardization and log-transformation of targets

### Model Training
- Hyperparameter optimization via grid search
- Cross-validation with stratified sampling
- Early stopping for neural networks

### Evaluation
- Multiple metrics: RMSE, MAE, R², accuracy, F1-score
- External validation on independent datasets
- Feature importance analysis and SHAP values

### Chemical Extrapolation
- Predict reactivity ratios for novel monomer combinations
- Chemical space analysis and similarity metrics
- Uncertainty quantification for predictions

## Output

Models generate:
- **Trained models** saved as pickle/joblib files
- **Performance metrics** in `.results.txt` files  
- **Predictions** for validation datasets
- **Feature importance** rankings

## Configuration

Key parameters typically configured:
- Feature sets (DFT vs RDKit descriptors)
- Capping strategies (methyl, hydrogen, methoxyl)
- Train/test splits and cross-validation folds
- Model hyperparameters

## Integration

Models are designed to work with:
- Descriptors from `descriptor_gen/` pipeline
- Datasets from `dataset/` directory
- Utilities from `utils/` module for feature engineering
