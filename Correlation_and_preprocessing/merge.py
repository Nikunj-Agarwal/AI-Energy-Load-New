import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path, time_col=None):
    """Load CSV and convert timestamp column to datetime index"""
    print(f"Loading {file_path}...")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"ERROR: File does not exist: {file_path}")
            return None
            
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Auto-detect time column if not specified
        if time_col is None:
            if 'timestamp' in df.columns:
                time_col = 'timestamp'
                print(f"Auto-detected time column: {time_col}")
            elif 'time_bucket' in df.columns:
                time_col = 'time_bucket'
                print(f"Auto-detected time column: {time_col}")
            elif 'datetime' in df.columns:
                time_col = 'datetime'
                print(f"Auto-detected time column: {time_col}")
            else:
                # Try to find any datetime-like column
                for col in df.columns:
                    try:
                        if pd.api.types.is_string_dtype(df[col]):
                            pd.to_datetime(df[col].iloc[0])
                            time_col = col
                            print(f"Found datetime-compatible column: {time_col}")
                            break
                    except:
                        continue
        
        if time_col and time_col in df.columns:
            # Convert time column to datetime and set as index
            df[time_col] = pd.to_datetime(df[time_col])
            
            # Keep a copy of datetime column before setting index
            if time_col != 'datetime':
                df['datetime'] = df[time_col]
                
            df.set_index(time_col, inplace=True)
            print(f"Set index to '{time_col}' with range: {df.index.min()} to {df.index.max()}")
            
            # Check for duplicate indices
            if df.index.duplicated().any():
                dup_count = df.index.duplicated().sum()
                print(f"WARNING: Found {dup_count} duplicate timestamps in {file_path}")
                
                # Keep last observation for each timestamp
                df = df[~df.index.duplicated(keep='last')]
                print(f"Removed duplicates - new shape: {df.shape}")
                
            return df
        else:
            print(f"ERROR: Time column '{time_col}' not found in dataframe")
            print(f"Available columns: {df.columns.tolist()}")
            return None
    
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")
        return None

def merge_datasets(gkg_df, weather_df, output_path):
    """Merge two dataframes on their datetime indices"""
    print("Merging datasets...")
    
    # Check if both dataframes have datetime indices
    if not (isinstance(gkg_df.index, pd.DatetimeIndex) and 
            isinstance(weather_df.index, pd.DatetimeIndex)):
        print("ERROR: Both dataframes must have datetime indices")
        return None
    
    # Get original shapes for reference
    gkg_shape = gkg_df.shape
    weather_shape = weather_df.shape
    
    # Sort indices to ensure proper alignment
    gkg_df = gkg_df.sort_index()
    weather_df = weather_df.sort_index()
    
    # Handle duplicate column names
    duplicates = set(gkg_df.columns).intersection(set(weather_df.columns))
    if duplicates:
        print(f"Handling {len(duplicates)} duplicate column names")
        for col in duplicates:
            if col != 'datetime':  # Preserve datetime column
                weather_df = weather_df.rename(columns={col: f"{col}_weather"})
    
    # Check for time range overlap
    overlap_start = max(gkg_df.index.min(), weather_df.index.min())
    overlap_end = min(gkg_df.index.max(), weather_df.index.max())
    
    if overlap_start > overlap_end:
        print("ERROR: No overlap between datasets' time ranges")
        print(f"GKG: {gkg_df.index.min()} to {gkg_df.index.max()}")
        print(f"Weather: {weather_df.index.min()} to {weather_df.index.max()}")
        return None
    
    print(f"Overlapping time range: {overlap_start} to {overlap_end}")
    print(f"GKG points in range: {len(gkg_df.loc[overlap_start:overlap_end])}")
    print(f"Weather points in range: {len(weather_df.loc[overlap_start:overlap_end])}")
    
    # Merge the dataframes
    merged_df = pd.merge(gkg_df, weather_df, 
                         left_index=True, right_index=True, 
                         how='inner')
    
    print(f"Merged dataframe shape: {merged_df.shape}")
    print(f"Retained {100*len(merged_df)/min(gkg_shape[0], weather_shape[0]):.1f}% of data after merge")
    
    # Handle missing values
    missing_counts = merged_df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if not cols_with_missing.empty:
        print(f"Columns with missing values: {len(cols_with_missing)}")
        for col, count in cols_with_missing.items():
            pct = 100 * count / len(merged_df)
            if pct > 1:  # Only show columns with >1% missing
                print(f"  {col}: {count} missing values ({pct:.1f}%)")
        
        print("Filling missing values...")
        # For time series data, forward fill is often better than mean
        merged_df = merged_df.ffill()
        # If still has NaN (at the beginning), fill with backward fill
        merged_df = merged_df.bfill()
    
    # Reset index to have datetime as a column
    merged_df = merged_df.reset_index()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the merged data
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")
    
    # Create visualization of merged data
    print("Creating merged data visualization...")
    create_merge_visualization(merged_df, os.path.dirname(output_path))
    
    return merged_df

def create_merge_visualization(df, output_dir):
    """Create and save a visualization of the merged data"""
    try:
        # Create figures directory
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Find energy consumption column
        energy_cols = [col for col in df.columns if 'load' in col.lower() or 'demand' in col.lower() or 'consumption' in col.lower()]
        
        if not energy_cols:
            print("WARNING: Could not find energy consumption column for visualization")
            return
            
        energy_col = energy_cols[0]
        print(f"Using '{energy_col}' as energy column for visualization")
        
        # Get a sample for visualization (last 2 weeks)
        if len(df) > 1344:  # 15-min intervals for 2 weeks = 96*14
            sample_df = df.iloc[-1344:].copy()
        else:
            sample_df = df.copy()
            
        # Find the datetime column - check various possible names
        datetime_col = None
        for col_name in ['datetime', 'datetime_y', 'index', 'timestamp', 'time_bucket']:
            if col_name in sample_df.columns:
                datetime_col = col_name
                print(f"Using '{datetime_col}' as datetime column for visualization")
                break
                
        if datetime_col is None:
            print("WARNING: No datetime column found for visualization")
            return
            
        # Ensure datetime is properly formatted
        sample_df[datetime_col] = pd.to_datetime(sample_df[datetime_col])
            
        # Plot energy and a few key features
        plt.figure(figsize=(15, 10))
        
        # Energy demand
        plt.subplot(3, 1, 1)
        plt.plot(sample_df[datetime_col], sample_df[energy_col], 'b-')
        plt.title('Energy Demand')
        plt.grid(True, alpha=0.3)
        
        # Find and plot a weather feature (like temperature)
        weather_cols = [col for col in df.columns if any(x in col.lower() for x in ['temp', 'wind', 'humid'])]
        if weather_cols:
            plt.subplot(3, 1, 2)
            plt.plot(sample_df[datetime_col], sample_df[weather_cols[0]], 'g-')
            plt.title(f'Weather Feature: {weather_cols[0]}')
            plt.grid(True, alpha=0.3)
            
        # Find and plot a news feature
        news_cols = [col for col in df.columns if any(x in col.lower() for x in ['theme', 'article', 'tone'])]
        if news_cols:
            plt.subplot(3, 1, 3)
            plt.plot(sample_df[datetime_col], sample_df[news_cols[0]], 'r-')
            plt.title(f'News Feature: {news_cols[0]}')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'merged_data_overview.png'), dpi=300)
        plt.close()
        
        print(f"Visualization saved to {figures_dir}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()  # Print full error trace

def main():
    # Define base directory paths - FIXED to match current system
    base_dir = Path(r'c:\Users\nikun\Desktop\MLPR\AI-Energy-Load-New')
    
    # Input file paths
    gkg_path = base_dir / 'OUTPUT_DIR' / 'aggregated_data' / 'aggregated_gkg_15min.csv'
    weather_path = base_dir / 'Weather_Energy' / 'weather_energy_15min.csv'
    
    # Try alternate weather paths if first doesn't exist
    if not os.path.exists(weather_path):
        alt_paths = [
            base_dir / 'Weather_Energy' / 'weather_energy_15min.csv',
            base_dir / 'weather_energy' / 'weather_energy_15min.csv',
            base_dir / 'Weather Energy' / 'weather_energy_15min.csv',  # With space
            base_dir / 'Weather' / 'weather_energy_15min.csv',
            base_dir / 'DATA' / 'Weather_Energy' / 'weather_energy_15min.csv'
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                weather_path = path
                print(f"Found weather data at alternate path: {weather_path}")
                break
                
    # Output file path
    output_dir = base_dir / 'OUTPUT_DIR' / 'merged_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'merged_gkg_weather_energy.csv'
    
    print(f"GKG data path: {gkg_path}")
    print(f"Weather data path: {weather_path}")
    print(f"Output path: {output_path}")
    
    # Load datasets
    gkg_df = load_data(gkg_path, time_col='time_bucket')
    weather_df = load_data(weather_path, time_col='timestamp')
    
    if gkg_df is None or weather_df is None:
        print("ERROR: Failed to load one or both datasets")
        if not os.path.exists(gkg_path):
            print(f"GKG file not found: {gkg_path}")
        if not os.path.exists(weather_path):
            print(f"Weather file not found: {weather_path}")
            print("Available directories in project root:")
            for item in os.listdir(base_dir):
                if os.path.isdir(base_dir / item):
                    print(f"  - {item}")
        return
    
    # Merge and save datasets
    merged_df = merge_datasets(gkg_df, weather_df, output_path)
    
    if merged_df is not None:
        print("Merge completed successfully")
        print(f"Final dataset contains {len(merged_df)} rows and {len(merged_df.columns)} columns")
        print("Sample of merged data:")
        print(merged_df.head(3))
        
        # Create summary file
        summary_path = output_dir / 'merge_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Merge Summary\n")
            f.write("=============\n\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"GKG data shape: {gkg_df.shape}\n")
            f.write(f"Weather data shape: {weather_df.shape}\n")
            f.write(f"Merged data shape: {merged_df.shape}\n\n")
            
            # Find datetime column for date range
            datetime_col = None
            for col_name in ['datetime', 'datetime_y', 'index', 'timestamp', 'time_bucket']:
                if col_name in merged_df.columns:
                    datetime_col = col_name
                    break
            
            if datetime_col:
                f.write(f"Date range: {pd.to_datetime(merged_df[datetime_col]).min()} to {pd.to_datetime(merged_df[datetime_col]).max()}\n\n")
            else:
                f.write("Date range: Unknown (no datetime column found)\n\n")
            
            # Count features by type
            news_cols = len([c for c in merged_df.columns if any(x in c.lower() for x in ['theme', 'tone', 'article'])])
            weather_cols = len([c for c in merged_df.columns if any(x in c.lower() for x in ['temp', 'wind', 'humid', 'precip'])])
            energy_cols = len([c for c in merged_df.columns if any(x in c.lower() for x in ['load', 'demand', 'consumption'])])
            
            f.write(f"Feature counts:\n")
            f.write(f"  News features: {news_cols}\n")
            f.write(f"  Weather features: {weather_cols}\n")
            f.write(f"  Energy features: {energy_cols}\n")
            f.write(f"  Other features: {len(merged_df.columns) - news_cols - weather_cols - energy_cols}\n")
        
        print(f"Merge summary saved to {summary_path}")

if __name__ == "__main__":
    main()