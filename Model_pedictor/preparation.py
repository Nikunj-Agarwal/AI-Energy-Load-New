import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

# Add the parent directory to the path to find the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from config
from config import PATHS, setup_and_verify

# Use PATHS from config rather than hardcoding them
MERGED_DATA_PATH = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\Correlation_and_preprocessing\merged_gkg_weather_energy.csv"
SPLIT_DATA_DIR = PATHS["SPLIT_DATA_DIR"]  # Use from config
MODEL_DIR = PATHS["MODELS_DIR"]           # Use from config

# Ensure directories exist
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Add these imports at the top of your file, after the existing imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import calendar

def load_data(file_path):
    """Load the merged dataset"""
    print(f"Loading data from {file_path}")
    try:
        # Load the CSV file with datetime index
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        print(f"Successfully loaded {len(df)} rows with {len(df.columns)} features")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("Handling missing values...")
    
    # Count missing values before
    missing_before = df.isna().sum().sum()
    print(f"Missing values before handling: {missing_before}")
    
    # First, forward fill within small gaps (up to 3 steps, 45 minutes)
    df = df.fillna(method='ffill', limit=3)
    
    # For columns with >5% missing values, use interpolation
    missing_pct = df.isna().mean()
    for col in df.columns:
        if missing_pct[col] > 0.05:
            print(f"Column {col} has {missing_pct[col]*100:.2f}% missing - using linear interpolation")
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    # For any remaining NaNs, use column mean
    df = df.fillna(df.mean())
    
    # Count missing values after
    missing_after = df.isna().sum().sum()
    print(f"Missing values after handling: {missing_after}")
    
    return df

def split_time_series_data(df, target_col='Power demand_sum', test_fraction=0.2, strategy='month_based'):
    """
    Split time series data using various strategies
    
    Parameters:
    -----------
    df : DataFrame with datetime index
    target_col : Target column to predict
    test_fraction : Fraction to use for testing
    strategy : One of ['month_based', 'fully_random', 'seasonal_block']
    """
    print(f"Using '{strategy}' split strategy with {test_fraction:.0%} test fraction")
    
    # Sort data chronologically
    df = df.sort_index()
    
    # Create year and month columns for grouping
    df = df.copy()  # Avoid modifying the original dataframe
    df['year'] = df.index.year
    df['month'] = df.index.month
    # Add season: 0=Winter, 1=Spring, 2=Summer, 3=Fall
    df['season'] = (df['month'] % 12 + 3) // 3 % 4
    
    # Get all unique years in the dataset
    years = sorted(df['year'].unique())
    print(f"Dataset spans {len(years)} years: {years}")
    
    # STRATEGY 1: Month-based sampling
    if strategy == 'month_based':
        # Create masks for train and test data
        train_mask = np.ones(len(df), dtype=bool)
        
        # For each year, randomly select months for testing
        np.random.seed(42)  # For reproducibility
        test_months_by_year = {}
        
        for year in years:
            # Get available months for this year
            available_months = sorted(df[df['year'] == year]['month'].unique())
            
            # Calculate how many months to select for testing
            n_test_months = max(1, int(len(available_months) * test_fraction))
            
            # Randomly select months for testing
            test_months = sorted(np.random.choice(available_months, size=n_test_months, replace=False))
            test_months_by_year[year] = test_months
            
            # Update mask for this year's test months
            year_month_mask = (df['year'] == year) & (df['month'].isin(test_months))
            train_mask[year_month_mask] = False
        
        # Print the selected test months by year
        print("Selected test months by year:")
        for year, months in test_months_by_year.items():
            print(f"  {year}: {[calendar.month_name[m] for m in months]}")
        
        # Split the data based on the masks
        train_df = df[train_mask]
        test_df = df[~train_mask]
        
        # Capture split metadata
        split_info = {
            'strategy': 'month_based_sampling',
            'test_months_by_year': test_months_by_year
        }
    
    # STRATEGY 2: Fully random sampling (disregards time continuity)
    elif strategy == 'fully_random':
        # Create a random mask with desired ratio
        np.random.seed(42)  # For reproducibility
        test_mask = np.random.rand(len(df)) < test_fraction
        
        train_df = df[~test_mask]
        test_df = df[test_mask]
        
        split_info = {
            'strategy': 'fully_random',
            'test_fraction': test_fraction,
            'test_count': test_mask.sum(),
            'training_count': (~test_mask).sum()
        }
        
        print(f"Random split created with {split_info['test_count']} test samples")
        
    # STRATEGY 3: Seasonal blocks (ensure each season is represented)
    elif strategy == 'seasonal_block':
        # Get combinations of year and season
        df['year_season'] = df['year'].astype(str) + "_" + df['season'].astype(str)
        year_seasons = df['year_season'].unique()
        
        # For each year, ensure we sample from each season
        np.random.seed(42)  # For reproducibility
        test_mask = np.zeros(len(df), dtype=bool)
        test_seasons = {}
        
        # Group by year and season
        for year in years:
            test_seasons[year] = {}
            
            # For each season in this year
            for season in sorted(df[df['year'] == year]['season'].unique()):
                season_data = df[(df['year'] == year) & (df['season'] == season)]
                
                if len(season_data) > 0:
                    # Calculate how many points to select from this season
                    n_test_points = int(len(season_data) * test_fraction)
                    
                    # Select consecutive block from middle of the season
                    start_idx = len(season_data) // 2 - n_test_points // 2
                    indices = season_data.index[start_idx:start_idx + n_test_points]
                    
                    # Mark these indices as test data
                    test_mask |= df.index.isin(indices)
                    
                    # Store season info
                    season_names = ['Winter', 'Spring', 'Summer', 'Fall']
                    test_seasons[year][season_names[season]] = n_test_points
        
        train_df = df[~test_mask]
        test_df = df[test_mask]
        
        split_info = {
            'strategy': 'seasonal_block',
            'test_seasons': test_seasons
        }
        
        # Print seasons
        print("Test data blocks by season:")
        for year, seasons in test_seasons.items():
            print(f"  {year}: {seasons}")
    
    else:
        raise ValueError(f"Unknown split strategy: {strategy}. Use 'month_based', 'fully_random', or 'seasonal_block'")
    
    # Remove helper columns before returning
    train_df = train_df.drop(['year', 'month', 'season'], axis=1, errors='ignore')
    if 'year_season' in train_df.columns:
        train_df = train_df.drop('year_season', axis=1)
        
    test_df = test_df.drop(['year', 'month', 'season'], axis=1, errors='ignore')
    if 'year_season' in test_df.columns:
        test_df = test_df.drop('year_season', axis=1)
    
    print(f"Training data: {len(train_df)} rows")
    print(f"Testing data: {len(test_df)} rows")
    
    # Create X (features) and y (target) for both sets
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test, split_info

def normalize_features(X_train, X_test, y_train, y_test):
    """Normalize features for better model performance"""
    print("Normalizing features...")
    
    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()
    
    # Fit and transform the training data
    X_train_scaled = feature_scaler.fit_transform(X_train)
    # Reshape y for scaling
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_train_scaled = target_scaler.fit_transform(y_train_reshaped).flatten()
    
    # Transform the test data using the same scalers
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_reshaped = y_test.values.reshape(-1, 1)
    y_test_scaled = target_scaler.transform(y_test_reshaped).flatten()
    
    # Save the scalers for later use
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(MODEL_DIR, 'target_scaler.pkl'))
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler

def save_split_data(X_train, y_train, X_test, y_test, train_idx, test_idx, split_info, strategy_name="default"):
    """Save the split datasets in separate train/test directories"""
    print(f"Saving split data for {strategy_name} strategy")
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create train and test subdirectories
    train_dir = os.path.join(SPLIT_DATA_DIR, "train_data")
    test_dir = os.path.join(SPLIT_DATA_DIR, "test_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Save training data
    train_df = pd.DataFrame(X_train, index=train_idx)
    train_df['target'] = y_train
    train_path = os.path.join(train_dir, f'train_data_{strategy_name}_{timestamp}.csv')
    train_df.to_csv(train_path)
    print(f"Training data saved to {train_path}")
    
    # Save testing data
    test_df = pd.DataFrame(X_test, index=test_idx)
    test_df['target'] = y_test
    test_path = os.path.join(test_dir, f'test_data_{strategy_name}_{timestamp}.csv')
    test_df.to_csv(test_path)
    print(f"Testing data saved to {test_path}")
    
    # Save metadata about the split
    meta_dir = os.path.join(SPLIT_DATA_DIR, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    
    with open(os.path.join(meta_dir, f'split_info_{strategy_name}_{timestamp}.txt'), 'w') as f:
        f.write(f"Data split performed on: {datetime.now()}\n")
        f.write(f"Split strategy: {strategy_name}\n\n")
        
        # Write detailed strategy info
        if isinstance(split_info, dict):
            for key, value in split_info.items():
                if key == 'test_months_by_year' and isinstance(value, dict):
                    f.write("\nTest months by year:\n")
                    for year, months in value.items():
                        f.write(f"  {year}: {[calendar.month_name[m] for m in months]}\n")
                elif key == 'test_seasons' and isinstance(value, dict):
                    f.write("\nTest seasons by year:\n")
                    for year, seasons in value.items():
                        f.write(f"  {year}: {seasons}\n")
                else:
                    f.write(f"{key}: {value}\n")
        else:
            f.write(f"Split info: {split_info}\n")
            
        f.write(f"\nTraining data: {len(train_df)} rows\n")
        f.write(f"Testing data: {len(test_df)} rows\n")
    
    # Also save split visualization
    create_split_visualization(train_idx, test_idx, y_train, y_test, split_info, meta_dir, timestamp, strategy_name)
    
    return train_path, test_path

def create_split_visualization(train_idx, test_idx, y_train, y_test, split_info, output_dir, timestamp, strategy_name):
    """Create and save visualizations of the data split"""
    
    # Create power demand plot
    plt.figure(figsize=(14, 7))
    
    # Plot all data points as small dots
    all_idx = train_idx.union(test_idx)
    all_y = pd.Series(np.nan, index=all_idx)
    all_y.loc[train_idx] = y_train
    all_y.loc[test_idx] = y_test
    
    plt.plot(all_idx, all_y, 'k.', alpha=0.1, markersize=1)
    
    # Plot training data
    plt.plot(train_idx, y_train, 'b.', alpha=0.5, label='Training Data')
    
    # Plot test data with higher visibility
    plt.plot(test_idx, y_test, 'r.', alpha=0.7, label='Testing Data')
    
    plt.title(f'Energy Load Training/Testing Split ({strategy_name})')
    plt.xlabel('Date')
    plt.ylabel('Power Demand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, f'split_visualization_{strategy_name}_{timestamp}.png')
    plt.savefig(viz_path, dpi=300)
    print(f"Visualization saved to {viz_path}")
    
    # Also create a monthly distribution visualization
    plt.figure(figsize=(12, 6))
    
    train_counts = train_idx.month.value_counts().sort_index()
    test_counts = test_idx.month.value_counts().sort_index()
    
    months = range(1, 13)
    month_names = [calendar.month_abbr[m] for m in months]
    
    train_values = [train_counts.get(m, 0) for m in months]
    test_values = [test_counts.get(m, 0) for m in months]
    
    bar_width = 0.35
    r1 = np.arange(len(months))
    r2 = [x + bar_width for x in r1]
    
    plt.bar(r1, train_values, width=bar_width, label='Training', color='blue', alpha=0.7)
    plt.bar(r2, test_values, width=bar_width, label='Testing', color='red', alpha=0.7)
    
    plt.xlabel('Month')
    plt.ylabel('Number of Samples')
    plt.title(f'Monthly Distribution of Training and Test Data ({strategy_name})')
    plt.xticks([r + bar_width/2 for r in range(len(months))], month_names)
    plt.legend()
    plt.tight_layout()
    
    # Save the monthly distribution visualization
    monthly_viz_path = os.path.join(output_dir, f'monthly_distribution_{strategy_name}_{timestamp}.png')
    plt.savefig(monthly_viz_path, dpi=300)
    print(f"Monthly distribution visualization saved to {monthly_viz_path}")

def create_sequences(X, y, seq_length=24):
    """
    Create sequences for LSTM input
    seq_length: number of time steps in each sequence (24 = 6 hours at 15-min intervals)
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        # Predict the next time step
        y_seq.append(y[i+seq_length])
        
    return np.array(X_seq), np.array(y_seq)

def main():
    """Main execution function"""
    print("=== Energy Load Prediction Data Preparation ===")
    
    # Ensure directories exist
    setup_and_verify()
    
    # Create additional directories for test/train data
    train_dir = os.path.join(SPLIT_DATA_DIR, "train_data")
    test_dir = os.path.join(SPLIT_DATA_DIR, "test_data")
    meta_dir = os.path.join(SPLIT_DATA_DIR, "metadata")
    
    for directory in [train_dir, test_dir, meta_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load merged data
    merged_df = load_data(MERGED_DATA_PATH)
    if merged_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Clean the data
    merged_df = handle_missing_values(merged_df)
    
    # Try all three split strategies
    strategies = ['month_based', 'fully_random', 'seasonal_block']
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Processing {strategy.upper()} split strategy")
        print(f"{'='*50}")
        
        # Split the data with current strategy
        X_train, y_train, X_test, y_test, split_info = split_time_series_data(
            merged_df, 
            test_fraction=0.2,
            strategy=strategy
        )
        
        # Save original indices before normalization
        train_idx = X_train.index
        test_idx = X_test.index
        
        # Normalize the data
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler = normalize_features(
            X_train, X_test, y_train, y_test
        )
        
        # Save the split data with strategy in filename
        train_path, test_path = save_split_data(
            X_train_scaled, y_train_scaled, 
            X_test_scaled, y_test_scaled,
            train_idx, test_idx, split_info,
            strategy_name=strategy
        )
        
        print(f"\n{strategy} data preparation completed!")
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Testing set: {X_test_scaled.shape[0]} samples")
        print(f"Feature count: {X_train_scaled.shape[1]} features")
    
    print("\nAll split strategies processed successfully!")

if __name__ == "__main__":
    main()