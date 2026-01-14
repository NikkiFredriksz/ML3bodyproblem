# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:46:06 2025

@author: olafk
"""

from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.model_selection import cross_val_score
import optuna
import optuna.visualization as vis
import plotly.io as pio

SEED = 42

# Load training and test data
def load_data(file_path_train, file_path_test):
    df_train = pd.read_excel(file_path_train)
    X_train = df_train.drop('OUTCOME', axis=1)
    Y_train = df_train['OUTCOME']
    
    df_test = pd.read_excel(file_path_test)
    X_test = df_test.drop('OUTCOME', axis=1)
    Y_test = df_test['OUTCOME']
    
    Y_train_binary = (Y_train == 3).astype(int)
    Y_test_binary = (Y_test == 3).astype(int)
    
    return X_train, Y_train_binary, X_test, Y_test_binary

def prepare_data(X_train, Y_train_binary, X_test, Y_test_binary, scale_data=True):
    # Convert to numpy arrays
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    Y_train_binary_np = Y_train_binary.values.ravel() if hasattr(Y_train_binary, 'values') else Y_train_binary.ravel()
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    Y_test_binary_np = Y_test_binary.values.ravel() if hasattr(Y_test_binary, 'values') else Y_test_binary.ravel()
    
    # Ensure X data is 2D
    if len(X_train_np.shape) == 1:
        X_train_np = X_train_np.reshape(-1, 1)
    if len(X_test_np.shape) == 1:
        X_test_np = X_test_np.reshape(-1, 1)
    
    # Apply Robust Scaling
    if scale_data:
        scaler = RobustScaler()
        X_train_np = scaler.fit_transform(X_train_np)
        X_test_np = scaler.transform(X_test_np)
        print("Applied RobustScaler to features")
    
    return X_train_np, Y_train_binary_np, X_test_np, Y_test_binary_np

# Optuna hyperparameter optimization function
def optimize_hyperparameters(X_train, Y_train_binary, X_test, Y_test_binary, n_trials=150):
    print(f"Starting Optuna hyperparameter optimization with {n_trials} trials...")
    
    X_train_np, Y_train_binary_np, X_test_np, Y_test_binary_np = prepare_data(X_train, Y_train_binary, X_test, Y_test_binary)
    
    # Calculate class weights
    class_counts = np.bincount(Y_train_binary_np)
    scale_pos_weight = class_counts[0] / class_counts[1]
    
    print(f"Class distribution: {class_counts}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    do_F1_optimisation = input("Do you want to do an F1 optimisation (y/yes): ").strip().lower()
    if do_F1_optimisation not in ['y', 'yes']:
    
        def objective(trial):
            # search parameters
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight * 0.01, scale_pos_weight * 3.0),
            }
            
            # Create and train model
            model = XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='logloss',
                early_stopping_rounds=50,
                random_state=SEED
            )
            
            # Fit with validation set
            model.fit(X_train_np, Y_train_binary_np, eval_set=[(X_test_np, Y_test_binary_np)], verbose=False)
            
            best_score = model.best_score
            return best_score
        objective_func = objective
        
    else:
        def objective_F1(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight * 0.01, scale_pos_weight * 3.0),
            }
            
            model = XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='logloss',
                early_stopping_rounds=50,
                random_state=SEED
            )
            
            model.fit(X_train_np, Y_train_binary_np, eval_set=[(X_test_np, Y_test_binary_np)], verbose=False)
            
            # Calculate F1-score for class 1 on validation set
            preds = model.predict(X_test_np)
            f1_class1 = f1_score(Y_test_binary_np, preds, pos_label=1)
            
            # Return negative F1 because Optuna minimizes
            return -f1_class1
        
        objective_func = objective_F1
    
    # Create and run the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_func, n_trials=n_trials)
    
    print("Optimization completed!")

    if do_F1_optimisation in ['y', 'yes']:
        print(f"Best F1-score: {-study.best_value:.4f}")
    else:
        print(f"Best logloss: {study.best_value:.4f}")
        
    print(f"Best hyperparameters: {study.best_params}")
    
    plot_optimization_results(study)
    
    return study

def plot_optimization_results(study):
    """Hyperparm optimisation visualisations"""
    
    print("Creating hyperparameter optimization visualizations...")
    
    # Get all trial results
    trials = study.trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) < 2:
        print("Not enough completed trials for visualization")
        return
    
    # Optimization history
    plt.figure(figsize=(10, 6))
    best_values = [study.best_trial.value]
    for i, trial in enumerate(completed_trials[1:], 1):
        best_values.append(min(best_values[-1], trial.value))
    
    plt.plot(range(len(completed_trials)), [t.value for t in completed_trials], 
             'o-', alpha=0.7, label='Trial values')
    plt.plot(range(len(best_values)), best_values, 'r-', linewidth=2, label='Best value')
    plt.xlabel('Trial Number')
    plt.ylabel('Loss (logloss)')
    plt.title('Optimization History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Get top 2 most important parameters
    try:
        importance_dict = optuna.importance.get_param_importances(study)
        top_params = list(importance_dict.keys())[:2]
        
        if len(top_params) >= 2:
            # Create 2D scatter plot for top 2 parameters
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Extract parameter values and losses
            param1_values = []
            param2_values = []
            losses = []
            
            for trial in completed_trials:
                if top_params[0] in trial.params and top_params[1] in trial.params:
                    param1_values.append(trial.params[top_params[0]])
                    param2_values.append(trial.params[top_params[1]])
                    losses.append(trial.value)
            
            # Scatter plot with color gradient
            scatter = ax1.scatter(param1_values, param2_values, c=losses, 
                                cmap='viridis', alpha=0.7, s=50)
            ax1.set_xlabel(top_params[0])
            ax1.set_ylabel(top_params[1])
            ax1.set_title(f'2D Parameter Space: {top_params[0]} vs {top_params[1]}')
            plt.colorbar(scatter, ax=ax1, label='Loss (logloss)')
            ax1.grid(True, alpha=0.3)
            
            # Mark the best point
            best_trial = study.best_trial
            if top_params[0] in best_trial.params and top_params[1] in best_trial.params:
                ax1.scatter(best_trial.params[top_params[0]], 
                          best_trial.params[top_params[1]], 
                          c='red', s=200, marker='*', edgecolors='black', 
                          label=f'Best (loss={best_trial.value:.3f})')
                ax1.legend()
            
            # Parameter importances
            param_names = list(importance_dict.keys())[:6]  # Top 6 parameters
            importances = [importance_dict[name] for name in param_names]
            
            ax2.barh(range(len(param_names)), importances, color='skyblue')
            ax2.set_yticks(range(len(param_names)))
            ax2.set_yticklabels(param_names)
            ax2.set_xlabel('Importance')
            ax2.set_title('Hyperparameter Importances')
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.show()
            
            print(f"2D plot created for: {top_params[0]} vs {top_params[1]}")
            print(f"Best point: {top_params[0]}={best_trial.params[top_params[0]]:.3f}, "
                  f"{top_params[1]}={best_trial.params[top_params[1]]:.3f}, "
                  f"loss={best_trial.value:.3f}")
            
        else:
            print("Not enough parameters for 2D plot")
            
    except Exception as e:
        print(f"Could not create parameter importance plots: {e}")

# Train final model with best hyperparameters
def train_final_model(X_train, Y_train_binary, X_test, Y_test_binary, best_params):
    print("Training final model with optimized hyperparameters...")
    
    X_train_np, Y_train_binary_np, X_test_np, Y_test_binary_np = prepare_data(X_train, Y_train_binary, X_test, Y_test_binary)
    
    # Create model with best parameters
    final_model = XGBClassifier(
        **best_params,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=SEED
    )
    
    # Fit model
    final_model.fit(
        X_train_np, Y_train_binary_np,
        eval_set=[(X_train_np, Y_train_binary_np), (X_test_np, Y_test_binary_np)],
        verbose=True
    )
    
    return final_model

# Visualisations
def create_vis(best_model, X_train, Y_train_binary, Y_test_binary, X_test, best_lr):
    """Create comprehensive visualizations for the best model"""
    
    Y_test_binary_np = Y_test_binary.values.ravel() if hasattr(Y_test_binary, 'values') else Y_test_binary.ravel()
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Make predictions
    preds = best_model.predict(X_test_np)
    pred_proba = best_model.predict_proba(X_test_np)
    
    # Training performance and feature importance
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle(f'XGBoost Model Analysis - Training (Best Learning Rate: {best_lr})', fontsize=16, weight='bold')
    
    # Plot training and validation loss
    results = best_model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    axes1[0, 0].plot(x_axis, results['validation_0']['logloss'], label='Train', linewidth=2, color='blue')
    axes1[0, 0].plot(x_axis, results['validation_1']['logloss'], label='Test', linewidth=2, color='red')
    axes1[0, 0].legend()
    axes1[0, 0].set_ylabel('Multi-class Log Loss')
    axes1[0, 0].set_xlabel('Boosting Rounds')
    axes1[0, 0].set_title('Learning Curves')
    axes1[0, 0].grid(True, alpha=0.3)
    
    # Feature importance
    feature_importance = best_model.feature_importances_
    feature_names = X_train.columns
    indices = np.argsort(feature_importance)[::-1][:10]
    
    axes1[0, 1].barh(range(len(indices)), feature_importance[indices], color='skyblue')
    axes1[0, 1].set_yticks(range(len(indices)))
    axes1[0, 1].set_yticklabels([feature_names[i] for i in indices])
    axes1[0, 1].set_xlabel('Feature Importance')
    axes1[0, 1].set_title('Top 10 Feature Importance')
    
    # Accuracy over time
    train_accuracy = [1 - x for x in results['validation_0']['logloss']]
    test_accuracy = [1 - x for x in results['validation_1']['logloss']]
    
    axes1[1, 0].plot(x_axis, train_accuracy, label='Train Accuracy', linewidth=2, color='blue')
    axes1[1, 0].plot(x_axis, test_accuracy, label='Test Accuracy', linewidth=2, color='red')
    axes1[1, 0].legend()
    axes1[1, 0].set_ylabel('Accuracy (1 - Loss)')
    axes1[1, 0].set_xlabel('Boosting Rounds')
    axes1[1, 0].set_title('Accuracy Over Time')
    axes1[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Prediction performance
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle(f'XGBoost Model Analysis - Predictions (Best Learning Rate: {best_lr})', fontsize=16, weight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test_binary_np, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[0, 0])
    axes2[0, 0].set_xlabel('Predicted')
    axes2[0, 0].set_ylabel('Actual')
    axes2[0, 0].set_title('Confusion Matrix')
    
    # Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes2[0, 1])
    axes2[0, 1].set_xlabel('Predicted')
    axes2[0, 1].set_ylabel('Actual')
    axes2[0, 1].set_title('Normalized Confusion Matrix')
    
    # Prediction distribution
    colors = ['blue', 'red', 'green', 'orange']
    for class_label in range(4):
        if len(Y_test_binary_np) > 0 and np.sum(Y_test_binary_np == class_label) > 0:
            axes2[1, 0].hist(pred_proba[Y_test_binary_np == class_label, class_label], 
                           alpha=0.7, label=f'Class {class_label}', bins=20, color=colors[class_label])
    
    axes2[1, 0].set_xlabel('Predicted Probability')
    axes2[1, 0].set_ylabel('Frequency')
    axes2[1, 0].set_title('Prediction Distribution by True Class')
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)
    
    # Class distribution in predictions
    unique, counts = np.unique(preds, return_counts=True)
    axes2[1, 1].bar(unique, counts, color=colors[:len(unique)], alpha=0.7)
    axes2[1, 1].set_xlabel('Predicted Class')
    axes2[1, 1].set_ylabel('Count')
    axes2[1, 1].set_title('Predicted Class Distribution')
    axes2[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Classification report
    print("Classification report")
    print(classification_report(Y_test_binary_np, preds))
    
    return fig1, fig2

def main():
    # Load training and test data
    file_path_train = input("Please input the path to your training data excel file: ").strip()
    file_path_test = input("Please input the path to your test data excel file: ").strip()
    X_train, Y_train_binary, X_test, Y_test_binary = load_data(file_path_train, file_path_test)
    
    # Optuna optimise hyperparameters
    n_trials = int(input("How many optimization trials? (default 150): ") or 150)
    study = optimize_hyperparameters(X_train, Y_train_binary, X_test, Y_test_binary, n_trials)
        
    # Train final model with best parameters
    best_model = train_final_model(X_train, Y_train_binary, X_test, Y_test_binary, study.best_params)
    best_lr = study.best_params['learning_rate']
        
    print(f"Using optimized model with learning rate: {best_lr:.3f}")
    
    # Create final visualizations for best model
    create_vis(best_model, X_train, Y_train_binary, Y_test_binary, X_test, best_lr)
    
    print(f"Best model with learning rate {best_lr} selected and visualized!")

if __name__=='__main__':
    main()