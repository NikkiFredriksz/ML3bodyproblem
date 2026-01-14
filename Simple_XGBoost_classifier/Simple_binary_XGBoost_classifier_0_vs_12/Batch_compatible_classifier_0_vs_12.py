# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:03:49 2025

@author: olafk
"""


from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, balanced_accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import optuna
import joblib
import argparse
import sys
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xgboost_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

SEED = 42

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='XGBoost hyperparameter optimization for multi-class classification')
    parser.add_argument('train_file', type=str, help='Path to training data Excel file')
    parser.add_argument('test_file', type=str, help='Path to test data Excel file')
    parser.add_argument('--n_trials', type=int, default=150, help='Number of Optuna optimization trials (default: 150)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results (default: results)')
    parser.add_argument('--model_name', type=str, default='xgboost_model', help='Base name for saved models (default: xgboost_model)')
    parser.add_argument('--study_name', type=str, default='xgboost_study', help='Name for Optuna study (default: xgboost_study)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs for Optuna (default: -1 for all cores)')
    parser.add_argument('--study_name', type=str, default='xgboost_study', help='Name for Optuna study (default: xgboost_study)')
    
    return parser.parse_args()

def load_data(file_path_train, file_path_test):
    """Load training and test data from Excel files."""
    logger.info(f"Loading training data from {file_path_train}")
    df_train = pd.read_excel(file_path_train)
    
    logger.info(f"Loading test data from {file_path_test}")
    df_test = pd.read_excel(file_path_test)
    
    # Filter out rows where OUTCOME == 3
    logger.info("Filtering out rows with OUTCOME == 3")
    df_train = df_train[df_train['OUTCOME'] != 3]
    df_test = df_test[df_test['OUTCOME'] != 3]
    
    X_train = df_train.drop('OUTCOME', axis=1)
    Y_train = df_train['OUTCOME']
    
    X_test = df_test.drop('OUTCOME', axis=1)
    Y_test = df_test['OUTCOME']

    Y_train = (Y_train == 0).astype(int)
    Y_test = (Y_test == 0).astype(int)

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Class distribution in training: {Y_train.value_counts().to_dict()}")
    logger.info(f"Class distribution in test: {Y_test.value_counts().to_dict()}")
    
    return X_train, Y_train, X_test, Y_test

def prepare_data(X_train, Y_train, X_test, Y_test):
    """Prepare data for XGBoost."""
    # Convert to numpy arrays
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    Y_train_np = Y_train.values.ravel() if hasattr(Y_train, 'values') else Y_train.ravel()
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    Y_test_np = Y_test.values.ravel() if hasattr(Y_test, 'values') else Y_test.ravel()
    
    # Ensure X data is 2D
    if len(X_train_np.shape) == 1:
        X_train_np = X_train_np.reshape(-1, 1)
    if len(X_test_np.shape) == 1:
        X_test_np = X_test_np.reshape(-1, 1)
    
    return X_train_np, Y_train_np, X_test_np, Y_test_np

def optimize_hyperparameters(X_train, Y_train, X_test, Y_test, n_trials=150, study_name='xgboost_study', n_jobs=-1, output_dir='results'):
    """Optuna hyperparameter optimization function."""
    logger.info(f"Starting Optuna hyperparameter optimization with {n_trials} trials")
    
    X_train_np, Y_train_np, X_test_np, Y_test_np = prepare_data(X_train, Y_train, X_test, Y_test)
    
    # Calculate class weights
    class_counts = np.bincount(Y_train_np)
    imbalance_ratio = class_counts[0] / class_counts[1]
    
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Imbalance ratio (0/1): {imbalance_ratio:.2f}")
    logger.info(f"Percentage of class 1: {100*class_counts[1]/len(Y_train_np):.1f}%")
    
    def objective(trial):
        # search parameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 750),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'gamma': trial.suggest_float('gamma', 0, 2),
            
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 
                                                   imbalance_ratio * 0.5,  # Start lower
                                                   imbalance_ratio * 3.0,  # Go higher
                                                   log=True),
        }

        
        model = XGBClassifier(
            **params,
            objective='binary:logistic',  
            eval_metric=['logloss', 'error', 'auc'],  
            early_stopping_rounds=50,  
            random_state=SEED,
            n_jobs=1,
            verbosity=0
        )
        
        # Use cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # More folds
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train_np, Y_train_np):
            X_cv_train, X_cv_val = X_train_np[train_idx], X_train_np[val_idx]
            y_cv_train, y_cv_val = Y_train_np[train_idx], Y_train_np[val_idx]
            
            base_weights = np.ones(len(y_cv_train))
            class_1_mask = (y_cv_train == 1)
            
            # Let Optuna optimize how much to emphasize class 1
            class_1_weight = trial.suggest_float('class_1_weight', 2.0, 10.0)
            base_weights[class_1_mask] = class_1_weight
            
            model.fit(
                X_cv_train, y_cv_train,
                sample_weight=base_weights,
                eval_set=[(X_cv_val, y_cv_val)],
                verbose=False
            )
            
            # Get probabilities for threshold tuning
            preds_proba = model.predict_proba(X_cv_val)[:, 1]
            
            # Optimize threshold for better class 1 recall
            threshold = trial.suggest_float('threshold', 0.05, 0.6)
            preds = (preds_proba >= threshold).astype(int)
            
            # Calculate class 1 recall
            recall_class1 = recall_score(y_cv_val, preds, pos_label=1, zero_division=0)
            
            # Also calculate class 1 F1
            f1_class1 = f1_score(y_cv_val, preds, pos_label=1, zero_division=0)
            
            # Weighted score: 70% recall, 30% F1 
            weighted_score = 0.7 * recall_class1 + 0.3 * f1_class1
            
            cv_scores.append(-weighted_score)
        
        return np.mean(cv_scores)
    
    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize'
    )
    
    logger.info(f"Starting optimization with {n_jobs} parallel jobs")
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Save study to file after optimization completes
    study_file = os.path.join(output_dir, f"{study_name}.pkl")
    joblib.dump(study, study_file)
    logger.info(f"Saved study to {study_file}")
    
    # Also save to CSV for easy analysis
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(output_dir, "optuna_trials_detailed.csv"), index=False)
    
    logger.info("Multi-class optimization completed!")
    logger.info(f"Best macro F1-score: {-study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    return study

def plot_optimization_results(study, output_dir='results'):
    """Create hyperparameter optimization visualizations and save to files."""
    logger.info("Creating hyperparameter optimization visualizations")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all trial results
    trials = study.trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) < 2:
        logger.warning("Not enough completed trials for visualization")
        return
    
    # 1. Optimization history (convergence plot)
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
    plt.savefig(os.path.join(output_dir, 'fig1_optimization_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved optimization history plot as fig1_optimization_history.png")
    
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
            
            # Parameter importances (bar chart)
            param_names = list(importance_dict.keys())[:6]  # Top 6 parameters
            importances = [importance_dict[name] for name in param_names]
            
            ax2.barh(range(len(param_names)), importances, color='skyblue')
            ax2.set_yticks(range(len(param_names)))
            ax2.set_yticklabels(param_names)
            ax2.set_xlabel('Importance')
            ax2.set_title('Hyperparameter Importances')
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig2_parameter_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved parameter analysis plot as fig2_parameter_analysis.png")
            
            logger.info(f"2D plot created for: {top_params[0]} vs {top_params[1]}")
            logger.info(f"Best point: {top_params[0]}={best_trial.params[top_params[0]]:.3f}, "
                      f"{top_params[1]}={best_trial.params[top_params[1]]:.3f}, "
                      f"loss={best_trial.value:.3f}")
            
        else:
            logger.warning("Not enough parameters for 2D plot")
            
    except Exception as e:
        logger.error(f"Could not create parameter importance plots: {e}")

def train_final_model(X_train, Y_train, X_test, Y_test, best_params):
    """Train final model with best hyperparameters."""
    logger.info("Training final model with optimized hyperparameters")
    
    X_train_np, Y_train_np, X_test_np, Y_test_np = prepare_data(X_train, Y_train, X_test, Y_test)
    
    # Extract and prepare class weights if present
    class_weights = {}
    params_to_remove = []
    for key in list(best_params.keys()):
        if key.startswith('weight_class_'):
            class_num = int(key.split('_')[-1])
            class_weights[class_num] = best_params[key]
            params_to_remove.append(key)
    
    # Remove weight parameters from best_params copy
    model_params = best_params.copy()
    for key in params_to_remove:
        model_params.pop(key, None)
    
    # Create model with best parameters
    final_model = XGBClassifier(
        **model_params,
        objective='binary:logistic',
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=150,
        random_state=SEED,
        n_jobs=-1 
    )
    
    # Apply class weights if they exist
    if class_weights:
        sample_weights = np.array([class_weights.get(y, 1.0) for y in Y_train_np])
        logger.info(f"Applying class weights: {class_weights}")
    else:
        sample_weights = None
    
    # Fit model
    logger.info("Starting model training...")
    final_model.fit(
        X_train_np, Y_train_np,
        sample_weight=sample_weights,
        eval_set=[(X_train_np, Y_train_np), (X_test_np, Y_test_np)],
        verbose=True
    )
    
    logger.info("Final model training completed")
    return final_model

def create_vis(best_model, X_train, Y_train, Y_test, X_test, best_lr, output_dir='results'):
    """Create comprehensive visualizations for the best model and save to files."""
    logger.info("Creating comprehensive visualizations for best model")
    
    Y_test_np = Y_test.values.ravel() if hasattr(Y_test, 'values') else Y_test.ravel()
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Make predictions
    preds = best_model.predict(X_test_np)
    pred_proba = best_model.predict_proba(X_test_np)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Training performance and feature importance
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle(f'XGBoost Model Analysis - Training (Best Learning Rate: {best_lr:.4f})', fontsize=16, weight='bold')
    
    # Plot training and validation loss
    results = best_model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    axes1[0, 0].plot(x_axis, results['validation_0']['logloss'], label='Train', linewidth=2, color='blue')
    axes1[0, 0].plot(x_axis, results['validation_1']['logloss'], label='Test', linewidth=2, color='red')
    axes1[0, 0].legend()
    axes1[0, 0].set_ylabel('Log Loss')
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
    
    # Save feature importance data
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved training analysis plot as fig3_training_analysis.png")
    
    # Prediction performance
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle(f'XGBoost Model Analysis - Predictions (Best Learning Rate: {best_lr:.4f})', fontsize=16, weight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test_np, preds)
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
        if len(Y_test_np) > 0 and np.sum(Y_test_np == class_label) > 0:
            axes2[1, 0].hist(pred_proba[Y_test_np == class_label, class_label], 
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
    plt.savefig(os.path.join(output_dir, 'fig4_prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved prediction analysis plot as fig4_prediction_analysis.png")
    
    # Save classification report
    report = classification_report(Y_test_np, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    logger.info("Classification report saved as classification_report.csv")
    logger.info("\n" + classification_report(Y_test_np, preds))
    
    return fig1, fig2

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Make study name unique with job ID if available
    if 'SLURM_JOB_ID' in os.environ:
        args.study_name = f"{args.study_name}_{os.environ['SLURM_JOB_ID']}"
    
    logger.info(f"Starting XGBoost optimization with seed {SEED}")
    logger.info(f"Training file: {args.train_file}")
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training and test data
    X_train, Y_train, X_test, Y_test = load_data(args.train_file, args.test_file)
    
    # Optuna optimize hyperparameters
    study = optimize_hyperparameters(
        X_train, Y_train, X_test, Y_test, 
        n_trials=args.n_trials,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir
    )
    
    # Save study results
    study_results = pd.DataFrame([t.params for t in study.trials])
    study_results['value'] = [t.value for t in study.trials]
    study_results.to_csv(os.path.join(args.output_dir, 'optuna_trials.csv'), index=False)
    
    # Save best parameters
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df.to_csv(os.path.join(args.output_dir, 'best_parameters.csv'), index=False)
    
    # Plot optimization results
    plot_optimization_results(study, args.output_dir)
    
    # Train final model with best parameters
    best_model = train_final_model(X_train, Y_train, X_test, Y_test, study.best_params)
    
    # Save the trained model
    model_path = os.path.join(args.output_dir, f'{args.model_name}.joblib')
    joblib.dump(best_model, model_path)
    logger.info(f"Saved trained model to {model_path}")
    
    # Create final visualizations for best model
    best_lr = study.best_params.get('learning_rate', 0.1)
    create_vis(best_model, X_train, Y_train, Y_test, X_test, best_lr, args.output_dir)
    
    # Save evaluation metrics
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    Y_test_np = Y_test.values.ravel() if hasattr(Y_test, 'values') else Y_test.ravel()
    
    preds = best_model.predict(X_test_np)
    final_f1 = f1_score(Y_test_np, preds, average='macro')
    
    metrics = {
        'macro_f1_score': final_f1,
        'best_loss': study.best_value,
        'n_trials': args.n_trials,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, 'final_metrics.csv'), index=False)
    
    logger.info("Optimization completed successfully!")
    logger.info(f"Best macro F1-score: {-study.best_value:.4f}")
    logger.info(f"Final model F1-score: {final_f1:.4f}")
    logger.info(f"All results saved to {args.output_dir}/")

if __name__ == '__main__':
    main()
