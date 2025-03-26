import pandas as pd
import os
from pathlib import Path
import numpy as np

def load_data(file_path, time_col=None):
    """Load CSV and convert timestamp column to datetime index"""
    print(f"Loading {file_path}...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows")
        
        # Auto-detect time column if not specified
        if time_col is None:
            if 'timestamp' in df.columns:
                time_col = 'timestamp'
            elif 'time_bucket' in df.columns:
                time_col = 'time_bucket'
            elif 'datetime' in df.columns:
                time_col = 'datetime'
            else:
                # Try to find any datetime-like column
                for col in df.columns:
                    try:
                        if pd.api.types.is_string_dtype(df[col]):
                            pd.to_datetime(df[col].iloc[0])
                            time_col = col
                            break
                    except:
                        continue
        
        if time_col and time_col in df.columns:
            # Convert time column to datetime and set as index
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
            print(f"Set index to '{time_col}' with range: {df.index.min()} to {df.index.max()}")
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
    
    # Sort indices to ensure proper alignment
    gkg_df = gkg_df.sort_index()
    weather_df = weather_df.sort_index()
    
    # Handle duplicate column names
    duplicates = set(gkg_df.columns).intersection(set(weather_df.columns))
    if duplicates:
        print(f"Handling {len(duplicates)} duplicate column names")
        for col in duplicates:
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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the merged data
    merged_df.to_csv(output_path)
    print(f"Merged data saved to {output_path}")
    
    return merged_df

def main():
    # Define base directory paths
    base_dir = Path(r'c:\Users\nikun\Desktop\MLPR\AI_Energy-Load')
    
    # Input file paths
    gkg_path = base_dir / 'OUTPUT_DIR' / 'aggregated_data' / 'aggregated_gkg_15min.csv'
    weather_path = base_dir / 'Weather_Energy' / 'weather_energy_15min.csv'
    
    # Output file path
    output_dir = base_dir / 'Correlation_and_preprocessing'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'merged_gkg_weather_energy.csv'
    
    # Load datasets
    gkg_df = load_data(gkg_path, time_col='time_bucket')
    weather_df = load_data(weather_path, time_col='timestamp')
    
    if gkg_df is None or weather_df is None:
        print("ERROR: Failed to load one or both datasets")
        return
    
    # Merge and save datasets
    merged_df = merge_datasets(gkg_df, weather_df, output_path)
    
    if merged_df is not None:
        print("Merge completed successfully")
        print(f"Final dataset contains {len(merged_df)} rows and {len(merged_df.columns)} columns")
        print("Sample of merged data:")
        print(merged_df.head(3))

if __name__ == "__main__":
    main()