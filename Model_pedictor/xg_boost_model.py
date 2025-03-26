import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to locate config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

# Define paths
SPLIT_DATA_DIR = PATHS["SPLIT_DATA_DIR"]
MODELS_DIR = PATHS["MODELS_DIR"]
EVAL_DIR = PATHS["MODEL_EVAL_DIR"]

# Ensure directories exist
os.makedirs(os.path.join(MODELS_DIR, 'xgboost'), exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def load_split_data(strategy="month_based"):
    """Load the most recent split data files for the specified strategy"""
    print(f"Loading split data for '{strategy}' strategy...")
    
    # Find most recent train and test files
    train_dir = os.path.join(SPLIT_DATA_DIR, "train_data")
    test_dir = os.path.join(SPLIT_DATA_DIR, "test_data")
    
    # Get list of files matching the strategy
    train_files = [f for f in os.listdir(train_dir) if f.startswith(f"train_data_{strategy}_")]
    test_files = [f for f in os.listdir(test_dir) if f.startswith(f"test_data_{strategy}_")]
    
    if not train_files or not test_files:
        raise FileNotFoundError(f"No split data files found for strategy '{strategy}'")
    
    # Sort by timestamp (newest first)
    train_files.sort(reverse=True)
    test_files.sort(reverse=True)
    
    # Load the most recent files
    train_path = os.path.join(train_dir, train_files[0])
    test_path = os.path.join(test_dir, test_files[0])
    
    print(f"Loading training data from: {train_files[0]}")
    print(f"Loading testing data from: {test_files[0]}")
    
    train_df = pd.read_csv(train_path, parse_dates=True, index_col=0)
    test_df = pd.read_csv(test_path, parse_dates=True, index_col=0)
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")
    
    return X_train, y_train, X_test, y_test

def add_time_features(X_train, X_test):
    """Add time-based features to help XGBoost capture temporal patterns"""
    print("Adding time-based features...")
    
    # Make copies to avoid modifying originals
    X_train_time = X_train.copy()
    X_test_time = X_test.copy()
    
    # Add hour of day
    X_train_time['hour'] = X_train_time.index.hour
    X_test_time['hour'] = X_test_time.index.hour
    
    # Add day of week (0=Monday, 6=Sunday)
    X_train_time['dayofweek'] = X_train_time.index.dayofweek
    X_test_time['dayofweek'] = X_test_time.index.dayofweek
    
    # Add month
    X_train_time['month'] = X_train_time.index.month
    X_test_time['month'] = X_test_time.index.month
    
    # Add day of year
    X_train_time['dayofyear'] = X_train_time.index.dayofyear
    X_test_time['dayofyear'] = X_test_time.index.dayofyear
    
    # Add is_weekend flag
    X_train_time['is_weekend'] = (X_train_time.index.dayofweek >= 5).astype(int)
    X_test_time['is_weekend'] = (X_test_time.index.dayofweek >= 5).astype(int)
    
    # Add quarter of day (0-3)
    X_train_time['quarter_of_day'] = X_train_time.index.hour // 6
    X_test_time['quarter_of_day'] = X_test_time.index.hour // 6
    
    # Create cyclical features for hour to capture daily patterns
    X_train_time['hour_sin'] = np.sin(2 * np.pi * X_train_time.index.hour / 24)
    X_train_time['hour_cos'] = np.cos(2 * np.pi * X_train_time.index.hour / 24)
    X_test_time['hour_sin'] = np.sin(2 * np.pi * X_test_time.index.hour / 24)
    X_test_time['hour_cos'] = np.cos(2 * np.pi * X_test_time.index.hour / 24)
    
    # Create cyclical features for day of year to capture yearly patterns
    X_train_time['day_of_year_sin'] = np.sin(2 * np.pi * X_train_time.index.dayofyear / 365)
    X_train_time['day_of_year_cos'] = np.cos(2 * np.pi * X_train_time.index.dayofyear / 365)
    X_test_time['day_of_year_sin'] = np.sin(2 * np.pi * X_test_time.index.dayofyear / 365)
    X_test_time['day_of_year_cos'] = np.cos(2 * np.pi * X_test_time.index.dayofyear / 365)
    
    print(f"Added time features. New feature count: {len(X_train_time.columns)}")
    
    return X_train_time, X_test_time

# Fix for train_xgboost_model function
def train_xgboost_model(X_train, y_train, X_val, y_val, model_name):
    """Train an XGBoost model for energy load prediction"""
    print("Training XGBoost model...")
    
    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'n_estimators': 1000,
        'eval_metric': 'rmse',  # MOVED HERE from fit() method
        'early_stopping_rounds': 50,
        'seed': 42
    }
    
    # Create evaluation list for early stopping
    eval_set = [(X_train.values, y_train.values), (X_val.values, y_val.values)]
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    
    print("Starting model training...")
    model.fit(
        X_train.values, y_train.values,
        eval_set=eval_set,
        # REMOVED eval_metric='rmse' from here
        verbose=100  # Print progress every 100 iterations
    )
    
    # Get best iteration and score
    best_iteration = model.best_iteration
    best_score = model.best_score
    
    print(f"Best iteration: {best_iteration}")
    print(f"Best validation RMSE: {best_score:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'xgboost', f'{model_name}.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance
    importance_path = os.path.join(MODELS_DIR, 'xgboost', f'{model_name}_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    
    # Print top 10 features
    print("\nTop 10 important features:")
    print(importance_df.head(10))
    
    return model, importance_df

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and generate visualizations"""
    print("Evaluating model performance...")
    
    # Create evaluation directory
    eval_dir = os.path.join(EVAL_DIR, f'xgboost_{model_name}')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test.values)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    }, index=X_test.index)
    
    # Add error metrics
    predictions_df['error'] = predictions_df['actual'] - predictions_df['predicted']
    predictions_df['abs_error'] = abs(predictions_df['error'])
    predictions_df['percent_error'] = (predictions_df['error'] / predictions_df['actual']) * 100
    
    # Save predictions
    predictions_path = os.path.join(eval_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path)
    
    # Save metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_error': predictions_df['error'].mean(),
        'median_error': predictions_df['error'].median(),
        'max_error': predictions_df['error'].max(),
        'min_error': predictions_df['error'].min(),
        'mean_abs_error': predictions_df['abs_error'].mean(),
        'mean_percent_error': predictions_df['percent_error'].mean(),
        'median_percent_error': predictions_df['percent_error'].median(),
    }
    
    metrics_path = os.path.join(eval_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print metrics summary
    print("\nModel Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Error: {metrics['mean_error']:.4f}")
    print(f"Mean Absolute Error: {metrics['mean_abs_error']:.4f}")
    print(f"Mean Percent Error: {metrics['mean_percent_error']:.4f}%")
    
    # Create visualizations
    create_evaluation_plots(predictions_df, eval_dir, model_name)
    
    return predictions_df, metrics

def create_evaluation_plots(predictions_df, eval_dir, model_name):
    """Create and save evaluation plots"""
    plots_dir = os.path.join(eval_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(14, 7))
    plt.plot(predictions_df.index, predictions_df['actual'], 'b-', label='Actual', alpha=0.7)
    plt.plot(predictions_df.index, predictions_df['predicted'], 'r-', label='Predicted', alpha=0.7)
    plt.title(f'Actual vs Predicted Energy Load - XGBoost {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Power Demand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'), dpi=300)
    
    # Plot 2: Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(predictions_df['error'], bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Error Distribution - XGBoost {model_name}')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'), dpi=300)
    
    # Plot 3: Error over Time
    plt.figure(figsize=(14, 7))
    plt.plot(predictions_df.index, predictions_df['error'], 'g-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Error over Time - XGBoost {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_over_time.png'), dpi=300)
    
    # Plot 4: Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 10))
    plt.scatter(predictions_df['actual'], predictions_df['predicted'], alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted Scatter - XGBoost {model_name}')
    plt.xlabel('Actual Power Demand')
    plt.ylabel('Predicted Power Demand')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted_scatter.png'), dpi=300)
    
    # Plot 5: Feature Importance (For XGBoost)
    if os.path.exists(os.path.join(MODELS_DIR, 'xgboost', f'{model_name}_feature_importance.csv')):
        importance_df = pd.read_csv(os.path.join(MODELS_DIR, 'xgboost', f'{model_name}_feature_importance.csv'))
        
        plt.figure(figsize=(12, 8))
        
        # Get top 20 features
        top_features = importance_df.head(20)
        
        # Create horizontal bar chart
        plt.barh(range(len(top_features)), top_features['Importance'], align='center')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.title('Top 20 Feature Importance - XGBoost')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300)
    
    print(f"Evaluation plots saved to {plots_dir}")

def main():
    """Main execution function"""
    print("==== Energy Load XGBoost Model Training ====")
    
    # Process each split strategy
    strategies = ['month_based', 'fully_random', 'seasonal_block']
    
    for strategy in strategies:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {strategy.upper()} split strategy")
            print(f"{'='*50}")
            
            # Load split data
            X_train, y_train, X_test, y_test = load_split_data(strategy)
            
            # Add time-based features to help XGBoost capture temporal patterns
            X_train_time, X_test_time = add_time_features(X_train, X_test)
            
            # Define model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_name = f"{strategy}_{timestamp}"
            
            # Split some training data for validation
            val_size = int(0.2 * len(X_train_time))
            X_val = X_train_time.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_final = X_train_time.iloc[:-val_size]
            y_train_final = y_train.iloc[:-val_size]
            
            # Train model
            model, importance_df = train_xgboost_model(
                X_train_final, y_train_final,
                X_val, y_val,
                model_name
            )
            
            # Evaluate model
            predictions_df, metrics = evaluate_model(
                model, 
                X_test_time, 
                y_test, 
                model_name
            )
            
            print(f"\n{strategy} XGBoost model training and evaluation completed!\n")
            
        except Exception as e:
            print(f"ERROR processing {strategy} strategy: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll XGBoost model training and evaluation completed!")

if __name__ == "__main__":
    main()