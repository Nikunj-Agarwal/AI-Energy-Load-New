import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to locate config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

# Print TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Configure GPU memory growth to avoid memory allocation errors
def configure_gpu():
    """Configure TensorFlow to use GPU with optimal settings"""
    try:
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("No GPU found. Running on CPU.")
            return False
            
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        
        # Configure memory growth for better GPU memory management
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        # Print GPU info
        gpu_devices = tf.config.get_visible_devices('GPU')
        for device in gpu_devices:
            gpu_details = tf.config.experimental.get_device_details(device)
            if 'compute_capability' in gpu_details:
                cc = gpu_details['compute_capability']
                print(f"GPU {device.name} - Compute Capability: {cc[0]}.{cc[1]}")
            
        print("GPU memory growth enabled.")
        
        # Enable mixed precision for faster training (if supported)
        if tf.config.list_physical_devices('GPU'):
            print("Enabling mixed precision training")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
        return True
    except Exception as e:
        print(f"GPU configuration error: {e}")
        print("Falling back to CPU.")
        return False

# Define paths
SPLIT_DATA_DIR = PATHS["SPLIT_DATA_DIR"]
MODELS_DIR = PATHS["MODELS_DIR"]
EVAL_DIR = PATHS["MODEL_EVAL_DIR"]

# Ensure directories exist
os.makedirs(os.path.join(MODELS_DIR, 'rnn'), exist_ok=True)
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
    X_train = train_df.drop('target', axis=1).values
    y_train = train_df['target'].values
    
    X_test = test_df.drop('target', axis=1).values
    y_test = test_df['target'].values
    
    # Get original indices for plotting
    train_idx = train_df.index
    test_idx = test_df.index
    
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")
    
    return X_train, y_train, X_test, y_test, train_idx, test_idx

def create_sequences(X, y, seq_length=24):
    """
    Create sequences for RNN input
    seq_length: number of time steps in each sequence (24 = 6 hours at 15-min intervals)
    """
    print(f"Creating sequences with length {seq_length}...")
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        # Predict the next time step
        y_seq.append(y[i+seq_length])
        
    return np.array(X_seq), np.array(y_seq)

def build_rnn_model(input_shape, rnn_units=64, dropout_rate=0.2):
    """Build SimpleRNN model for energy load prediction"""
    model = Sequential([
        # First RNN layer with return sequences for stacking
        SimpleRNN(rnn_units, activation='tanh', return_sequences=True, 
                 input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second RNN layer
        SimpleRNN(rnn_units//2, activation='tanh'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer
        Dense(1)
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_name, epochs=100, batch_size=None):
    """Train the RNN model with early stopping"""
    print("Starting model training...")
    
    # Adjust batch size based on GPU availability
    if batch_size is None:
        if tf.config.list_physical_devices('GPU'):
            # Larger batch sizes work better on GPU
            batch_size = 64
        else:
            # Smaller batch sizes for CPU
            batch_size = 32
    
    print(f"Using batch size: {batch_size}")
    
    # Define input shape for model
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    
    # Build model
    model = build_rnn_model(input_shape)
    model.summary()
    
    # Create model checkpoint directory
    checkpoint_dir = os.path.join(MODELS_DIR, 'rnn', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model_path = os.path.join(MODELS_DIR, 'rnn', f'{model_name}.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history

def evaluate_model(model, X_test_seq, y_test_seq, test_idx, model_name):
    """Evaluate model performance and generate visualizations"""
    print("Evaluating model performance...")
    
    # Create evaluation directory
    eval_dir = os.path.join(EVAL_DIR, f'rnn_{model_name}')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Make predictions
    y_pred_seq = model.predict(X_test_seq)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_seq, y_pred_seq)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_seq, y_pred_seq)
    r2 = r2_score(y_test_seq, y_pred_seq)
    
    # Map predictions back to original timestamps (need to account for sequence length)
    seq_length = X_test_seq.shape[1]
    test_dates = test_idx[seq_length:]
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'actual': y_test_seq.flatten(),
        'predicted': y_pred_seq.flatten()
    }, index=test_dates)
    
    # Add error metrics
    predictions_df['error'] = predictions_df['actual'] - predictions_df['predicted']
    predictions_df['abs_error'] = abs(predictions_df['error'])
    predictions_df['percent_error'] = (predictions_df['error'] / predictions_df['actual']) * 100
    
    # Save predictions
    predictions_path = os.path.join(eval_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path)
    
    # Save metrics
    metrics = {
        'mse': float(mse),  # Convert to float for JSON serialization
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mean_error': float(predictions_df['error'].mean()),
        'median_error': float(predictions_df['error'].median()),
        'max_error': float(predictions_df['error'].max()),
        'min_error': float(predictions_df['error'].min()),
        'mean_abs_error': float(predictions_df['abs_error'].mean()),
        'mean_percent_error': float(predictions_df['percent_error'].mean()),
        'median_percent_error': float(predictions_df['percent_error'].median()),
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
    plt.title(f'Actual vs Predicted Energy Load - RNN {model_name}')
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
    plt.title(f'Error Distribution - RNN {model_name}')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'), dpi=300)
    
    # Plot 3: Error over Time
    plt.figure(figsize=(14, 7))
    plt.plot(predictions_df.index, predictions_df['error'], 'g-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Error over Time - RNN {model_name}')
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
    
    plt.title(f'Actual vs Predicted Scatter - RNN {model_name}')
    plt.xlabel('Actual Power Demand')
    plt.ylabel('Predicted Power Demand')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted_scatter.png'), dpi=300)
    
    print(f"Evaluation plots saved to {plots_dir}")

def plot_training_history(history, model_name, eval_dir):
    """Plot and save training history graphs"""
    plots_dir = os.path.join(eval_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot MAE curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300)
    print(f"Training history plot saved to {plots_dir}")

def main():
    """Main execution function"""
    print("==== Energy Load RNN Model Training ====")
    
    # Configure GPU acceleration
    has_gpu = configure_gpu()
    if has_gpu:
        print("Training will use GPU acceleration")
    else:
        print("Training will use CPU only - this may be slow")
    
    # Define sequence length for RNN (6 hours of 15-min intervals)
    seq_length = 24
    
    # Process each split strategy
    strategies = ['month_based', 'fully_random', 'seasonal_block']
    
    for strategy in strategies:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {strategy.upper()} split strategy")
            print(f"{'='*50}")
            
            # Load split data
            X_train, y_train, X_test, y_test, train_idx, test_idx = load_split_data(strategy)
            
            # Create sequences for RNN
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
            
            print(f"Sequence shapes - X_train: {X_train_seq.shape}, X_test: {X_test_seq.shape}")
            
            # Define model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_name = f"{strategy}_{timestamp}"
            
            # Split some training data for validation
            val_size = int(0.2 * len(X_train_seq))
            X_val_seq = X_train_seq[-val_size:]
            y_val_seq = y_train_seq[-val_size:]
            X_train_seq = X_train_seq[:-val_size]
            y_train_seq = y_train_seq[:-val_size]
            
            # Train model
            model, history = train_model(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                model_name,
                epochs=100
            )
            
            # Create evaluation directory for training history
            eval_dir = os.path.join(EVAL_DIR, f'rnn_{model_name}')
            os.makedirs(eval_dir, exist_ok=True)
            
            # Plot training history
            plot_training_history(history, model_name, eval_dir)
            
            # Evaluate model
            predictions_df, metrics = evaluate_model(
                model, 
                X_test_seq, 
                y_test_seq,
                test_idx, 
                model_name
            )
            
            print(f"\n{strategy} RNN model training and evaluation completed!\n")
            
        except Exception as e:
            print(f"ERROR processing {strategy} strategy: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll RNN model training and evaluation completed!")

if __name__ == "__main__":
    main()