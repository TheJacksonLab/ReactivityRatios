import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def get_xgb_model_regressor(xgb_params={}, random_state=42, 
                            eval_metric='rmse', 
                            objective='reg:squarederror', 
                            n_jobs=-1):
    if len(xgb_params) == 0:
        xgb_params = {'colsample_bytree': 0.9,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 2,
        'subsample': 0.9}
    model = XGBRegressor(
        random_state=random_state,
        eval_metric=eval_metric,
        objective=objective,
        n_jobs=n_jobs,
        **xgb_params
    )
    return model

def get_ANN_model_regressor(input_dim, output_dim, hidden_dim=128, 
                  num_layers=3, activation='relu'):
    activation_dict = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), "sigmoid": nn.Sigmoid()}
    activation_fn = activation_dict[activation]

    model = nn.Sequential()
    model.add_module('input_layer', nn.Linear(input_dim, hidden_dim))
    model.add_module('activation', activation_fn)
    for _ in range(num_layers - 1):
        model.add_module('hidden_layer', nn.Linear(hidden_dim, hidden_dim))
        model.add_module('activation', activation_fn)

    model.add_module('output_layer', nn.Linear(hidden_dim, output_dim))

    model.to(device)

    return model    
    
def train_xgb_model_regressor(X_train, y_train, params={}, model=None, random_state=42, 
                            eval_metric='rmse', 
                            objective='reg:squarederror', 
                            n_jobs=-1):
    if model is None:
        model = get_xgb_model_regressor(params, random_state, eval_metric, objective, n_jobs)
    model.fit(X_train, y_train)
    return model

def train_ANN_model_regressor(X_train, y_train, params={}, criterion=nn.MSELoss(), 
                    optimizer=optim.Adam, learning_rate=0.001, epochs=500, device=device, 
                    model=None, random_state=42, batch_size=32, 
                    X_val=None, y_val=None, early_stopping_patience=50, verbose=True):
    """
    Train an ANN regression model with batching, validation, and early stopping.
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: Parameters for the model creation
        criterion: Loss function
        optimizer: Optimizer class
        learning_rate: Learning rate for optimizer
        epochs: Maximum number of training epochs
        device: Device to train on ('cpu', 'cuda', or 'mps')
        model: Pre-initialized model (optional)
        random_state: Random seed for reproducibility
        activation: Activation function to use
        batch_size: Batch size for training
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        early_stopping_patience: Number of epochs to wait before early stopping (optional)
        verbose: Whether to print progress
    
    Returns:
        Trained model and dictionary of training history
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Initialize model if not provided
    if model is None:
        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        model = get_ANN_model_regressor(input_dim, output_dim, **params)
    
    # Initialize optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    
    # Convert data to tensors if they aren't already
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train).to(device)
    if not isinstance(y_train, torch.Tensor):
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        y_train = torch.FloatTensor(y_train).to(device)
    
    # Prepare validation data if provided
    has_validation = X_val is not None and y_val is not None
    if has_validation:
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.FloatTensor(X_val).to(device)
        if not isinstance(y_val, torch.Tensor):
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            y_val = torch.FloatTensor(y_val).to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [] if has_validation else None
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Create data loader for batch training
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Batch training
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation step
        if has_validation:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                history['val_loss'].append(val_loss)
                
                # Early stopping check
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            model.load_state_dict(best_model_state)
                            return model, history
        
        # Print progress
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            val_msg = f", Val Loss: {val_loss:.6f}" if has_validation else ""
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}{val_msg}")
    
    # Restore best model if early stopping was enabled but not triggered
    if has_validation and early_stopping_patience is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history
        
def evaluate_ANN_model_regressor(model, X_test, y_test, criterion=nn.MSELoss(), device=device, batch_size=None):
    """
    Evaluate ANN regression model on test data.
    
    Args:
        model: Trained ANN model
        X_test: Test features
        y_test: Test targets
        criterion: Loss function
        device: Device to evaluate on
        batch_size: Batch size for evaluation (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Convert data to tensors if they aren't already
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test).to(device)
    if not isinstance(y_test, torch.Tensor):
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        y_test = torch.FloatTensor(y_test).to(device)
    
    with torch.no_grad():
        if batch_size is None:
            # Evaluate all at once
            outputs = model(X_test)
            loss = criterion(outputs, y_test).item()
            
            # Convert to numpy for additional metrics
            y_pred = outputs.cpu().numpy()
            y_true = y_test.cpu().numpy()
        else:
            # Batch evaluation for larger datasets
            dataset = torch.utils.data.TensorDataset(X_test, y_test)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            
            losses = []
            y_preds = []
            y_trues = []
            
            for X_batch, y_batch in dataloader:
                batch_outputs = model(X_batch)
                losses.append(criterion(batch_outputs, y_batch).item())
                y_preds.append(batch_outputs.cpu().numpy())
                y_trues.append(y_batch.cpu().numpy())
            
            loss = sum(losses) / len(losses)
            y_pred = np.vstack(y_preds)
            y_true = np.vstack(y_trues)
    
    # Calculate additional metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'loss': loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
def evaluate_xgb_model_regressor(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        outputs = model.predict(X_test_tensor)
        loss = mean_squared_error(outputs, y_test_tensor)
        
    return loss
        
        
def get_xgb_model_classifier(xgb_params={}):
    if len(xgb_params) == 0:
        xgb_params = {'colsample_bytree': 0.9,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 2,
        'subsample': 0.9}
    model = XGBClassifier(**xgb_params)
    return model

def get_ANN_model_classifier(input_dim, output_dim, hidden_dim=128, 
                  num_layers=3, activation='relu'):
    activation_dict = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), "sigmoid": nn.Sigmoid()}
    activation_fn = activation_dict[activation]

    model = nn.Sequential()
    model.add_module('input_layer', nn.Linear(input_dim, hidden_dim))
    model.add_module('activation', activation_fn)
    for _ in range(num_layers - 1):
        model.add_module('hidden_layer', nn.Linear(hidden_dim, hidden_dim))
        model.add_module('activation', activation_fn)

    model.add_module('output_layer', nn.Linear(hidden_dim, output_dim))

    model.to(device)

    return model

def train_xgb_model_classifier(X_train, y_train, params={}, model=None):
    if model is None:
        model = get_xgb_model_classifier(params)
    model.fit(X_train, y_train)
    return model

def train_ANN_model_classifier(X_train, y_train, params={}, criterion=nn.MSELoss(), 
                    optimizer=optim.Adam, learning_rate=0.001, epochs=100, device=device, model=None):
    if model is None:
        model = get_ANN_model_classifier(X_train.shape[1], y_train.shape[1], **params)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
    return model

def evaluate_ANN_model_classifier(model, X_test, y_test, criterion=nn.MSELoss(), device=device):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        outputs = model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)
        
    return loss.item()

def evaluate_xgb_model_classifier(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        outputs = model.predict(X_test_tensor)
        loss = mean_squared_error(outputs, y_test_tensor)
        
    return loss 

# Linear Regression models
def get_linear_model_regressor(fit_intercept=True, n_jobs=None):
    """
    Create a Linear Regression model
    
    Args:
        fit_intercept: Whether to calculate the intercept for this model
        n_jobs: Number of jobs to use for the computation
        random_state: Random seed for reproducibility
    
    Returns:
        Linear Regression model
    """
    model = LinearRegression(
        fit_intercept=fit_intercept,
        n_jobs=n_jobs
    )
    return model

def train_linear_model_regressor(X_train, y_train, params={}, model=None):
    """
    Train a Linear Regression model
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: Parameters for the model creation
        model: Pre-initialized model (optional)
    
    Returns:
        Trained Linear Regression model
    """
    if model is None:
        model = get_linear_model_regressor(**params)
    
    # Reshape y_train if it's 1D
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    
    model.fit(X_train, y_train)
    return model

def evaluate_linear_model_regressor(model, X_test, y_test):
    """
    Evaluate Linear Regression model on test data
    
    Args:
        model: Trained Linear Regression model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Reshape y_test if it's 1D
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def get_logistic_model_classifier(penalty='l2', C=1.0, solver='lbfgs', multi_class='auto', 
                                 max_iter=100, random_state=42, n_jobs=None):
    """
    Create a Logistic Regression model
    
    Args:
        penalty: Norm used in the penalization
        C: Inverse of regularization strength
        solver: Algorithm to use in the optimization problem
        multi_class: How to handle multiple classes
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        n_jobs: Number of CPU cores used when parallelizing over classes
    
    Returns:
        Logistic Regression model
    """
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        multi_class=multi_class,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs
    )
    return model

def train_logistic_model_classifier(X_train, y_train, params={}, model=None):
    """
    Train a Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training targets (class labels)
        params: Parameters for the model creation
        model: Pre-initialized model (optional)
    
    Returns:
        Trained Logistic Regression model
    """
    if model is None:
        model = get_logistic_model_classifier(**params)
    
    model.fit(X_train, y_train)
    return model

def evaluate_logistic_model_classifier(model, X_test, y_test):
    """
    Evaluate Logistic Regression model on test data
    
    Args:
        model: Trained Logistic Regression model
        X_test: Test features
        y_test: Test targets (class labels)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle multi-class classification
    if len(np.unique(y_test)) > 2:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_report(y_test, y_pred)
    }

# SVM models
def get_svm_model_regressor(kernel='rbf', degree=3, gamma='scale', C=1.0, 
                           epsilon=0.1, tol=1e-3, max_iter=-1, random_state=42):
    """
    Create an SVM regression model
    
    Args:
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        degree: Degree of polynomial kernel
        gamma: Kernel coefficient
        C: Regularization parameter
        epsilon: Epsilon in the epsilon-SVR model
        tol: Tolerance for stopping criterion
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
    
    Returns:
        SVM regression model
    """
    model = SVR(
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        C=C,
        epsilon=epsilon,
        tol=tol,
        max_iter=max_iter
    )
    return model

def train_svm_model_regressor(X_train, y_train, params={}, model=None):
    """
    Train an SVM regression model
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: Parameters for the model creation
        model: Pre-initialized model (optional)
    
    Returns:
        Trained SVM regression model
    """
    if model is None:
        model = get_svm_model_regressor(**params)
    
    # Reshape y_train if needed
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        raise ValueError("SVR does not support multi-output regression. Use multiple SVR models instead.")
    elif len(y_train.shape) > 1:
        y_train = y_train.ravel()
    
    model.fit(X_train, y_train)
    return model

def evaluate_svm_model_regressor(model, X_test, y_test):
    """
    Evaluate SVM regression model on test data
    
    Args:
        model: Trained SVM regression model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Reshape y_test if needed
    if len(y_test.shape) > 1:
        y_test = y_test.ravel()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def get_svm_model_classifier(kernel='rbf', degree=3, gamma='scale', C=1.0, 
                            tol=1e-3, max_iter=-1, probability=True, random_state=42):
    """
    Create an SVM classification model
    
    Args:
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        degree: Degree of polynomial kernel
        gamma: Kernel coefficient
        C: Regularization parameter
        tol: Tolerance for stopping criterion
        max_iter: Maximum number of iterations
        probability: Enable probability estimates
        random_state: Random seed for reproducibility
    
    Returns:
        SVM classification model
    """
    model = SVC(
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        C=C,
        tol=tol,
        max_iter=max_iter,
        probability=probability,
        random_state=random_state
    )
    return model

def train_svm_model_classifier(X_train, y_train, params={}, model=None):
    """
    Train an SVM classification model
    
    Args:
        X_train: Training features
        y_train: Training targets (class labels)
        params: Parameters for the model creation
        model: Pre-initialized model (optional)
    
    Returns:
        Trained SVM classification model
    """
    if model is None:
        model = get_svm_model_classifier(**params)
    
    model.fit(X_train, y_train)
    return model

def evaluate_svm_model_classifier(model, X_test, y_test):
    """
    Evaluate SVM classification model on test data
    
    Args:
        model: Trained SVM classification model
        X_test: Test features
        y_test: Test targets (class labels)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle multi-class classification
    if len(np.unique(y_test)) > 2:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_report(y_test, y_pred)
    }

# Decision Tree models
def get_dt_model_regressor(params, random_state=42):
    """
    Create a Decision Tree regression model
    
    Args:
        criterion: Function to measure the quality of a split
        splitter: Strategy to choose the split at each node
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        random_state: Random seed for reproducibility
    
    Returns:
        Decision Tree regression model
    """
    model = DecisionTreeRegressor(
        **params,
        random_state=random_state
    )
    return model

def train_dt_model_regressor(X_train, y_train, params={}, model=None, random_state=42):
    """
    Train a Decision Tree regression model
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: Parameters for the model creation
        model: Pre-initialized model (optional)
    
    Returns:
        Trained Decision Tree regression model
    """
    if model is None:
        model = get_dt_model_regressor(params, random_state=random_state)
    
    # Reshape y_train if it's 1D
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    
    model.fit(X_train, y_train)
    return model

def evaluate_dt_model_regressor(model, X_test, y_test):
    """
    Evaluate Decision Tree regression model on test data
    
    Args:
        model: Trained Decision Tree regression model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Reshape y_test if it's 1D
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def get_dt_model_classifier(criterion='gini', splitter='best', max_depth=None, 
                           min_samples_split=2, min_samples_leaf=1, random_state=42):
    """
    Create a Decision Tree classification model
    
    Args:
        criterion: Function to measure the quality of a split
        splitter: Strategy to choose the split at each node
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        random_state: Random seed for reproducibility
    
    Returns:
        Decision Tree classification model
    """
    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    return model

def train_dt_model_classifier(X_train, y_train, params={}, model=None):
    """
    Train a Decision Tree classification model
    
    Args:
        X_train: Training features
        y_train: Training targets (class labels)
        params: Parameters for the model creation
        model: Pre-initialized model (optional)
    
    Returns:
        Trained Decision Tree classification model
    """
    if model is None:
        model = get_dt_model_classifier(**params)
    
    model.fit(X_train, y_train)
    return model

def evaluate_dt_model_classifier(model, X_test, y_test):
    """
    Evaluate Decision Tree classification model on test data
    
    Args:
        model: Trained Decision Tree classification model
        X_test: Test features
        y_test: Test targets (class labels)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle multi-class classification
    if len(np.unique(y_test)) > 2:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_report(y_test, y_pred)
    } 
