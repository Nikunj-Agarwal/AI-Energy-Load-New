import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_data(file_path):
    """
    Load the weather and energy data from CSV file
    """
    print(f"Loading data from: {file_path}")
    
    # First check which datetime column exists in the file
    try:
        # Read just the header to determine column names
        column_names = pd.read_csv(file_path, nrows=0).columns.tolist()
        
        # Determine which datetime column is present
        datetime_col = None
        if 'timestamp' in column_names:
            datetime_col = 'timestamp'
        elif 'datetime' in column_names:
            datetime_col = 'datetime'
        else:
            # If neither column exists, return data without parsing dates
            print("Warning: No datetime or timestamp column found. Dates will not be parsed.")
            return pd.read_csv(file_path)
        
        # Read the file with appropriate datetime parsing
        return pd.read_csv(file_path, parse_dates=[datetime_col])
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def aggregate_to_15min(df):
    """
    Convert 5-minute granularity data to 15-minute granularity based on specific column headers
    """
    # Ensure timestamp is datetime
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' not in df.columns:
        raise ValueError("Dataset must contain a 'datetime' or 'timestamp' column")
    
    if not pd.api.types.is_datetime64_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create 15-minute intervals - floor to nearest 15 min
    df['15min_interval'] = df['timestamp'].dt.floor('15min')
    
    # Define aggregation dictionary based on known column names
    agg_dict = {}
    
    # Energy/power columns
    if 'Power demand' in df.columns:
        agg_dict['Power demand'] = 'sum'
    
    # Temperature and related columns
    if 'temp' in df.columns:
        agg_dict['temp'] = ['mean', 'min', 'max']
    if 'dwpt' in df.columns:  # dew point
        agg_dict['dwpt'] = 'mean'
    if 'rhum' in df.columns:  # relative humidity
        agg_dict['rhum'] = 'mean'
    
    # Wind columns
    if 'wdir' in df.columns:  # wind direction
        agg_dict['wdir'] = 'mean'
    if 'wspd' in df.columns:  # wind speed
        agg_dict['wspd'] = ['mean', 'max']
    
    # Pressure
    if 'pres' in df.columns:
        agg_dict['pres'] = 'mean'
    
    # Skip time component columns as they're redundant with timestamp
    time_columns = ['year', 'month', 'day', 'hour', 'minute']
    
    # Moving averages should typically be recalculated after aggregation
    moving_avg_columns = [col for col in df.columns if 'moving_avg' in col]
    
    # Add any remaining numeric columns not explicitly handled
    for col in df.columns:
        if (col not in agg_dict and col not in ['datetime', 'timestamp', '15min_interval'] 
            and col not in time_columns and col not in moving_avg_columns
            and pd.api.types.is_numeric_dtype(df[col])):
            agg_dict[col] = 'mean'
    
    # Group by 15-minute intervals and aggregate
    df_15min = df.groupby('15min_interval').agg(agg_dict)
    
    # Flatten multi-level columns if they exist
    if isinstance(df_15min.columns, pd.MultiIndex):
        df_15min.columns = ['_'.join(col).strip() for col in df_15min.columns.values]
    
    # Reset index to make 15min_interval a column again
    df_15min = df_15min.reset_index()
    df_15min = df_15min.rename(columns={'15min_interval': 'timestamp'})
    
    # Recalculate moving averages if needed
    if 'Power demand' in df_15min.columns and 'moving_avg_3' in df.columns:
        df_15min['moving_avg_3'] = df_15min['Power demand'].rolling(window=3, min_periods=1).mean()
    
    return df_15min

def save_aggregated_data(df, output_path):
    """
    Save the aggregated data to CSV
    """
    print(f"Saving aggregated data to: {output_path}")
    df.to_csv(output_path, index=False)

def main():
    # Fix: Point to the directory containing data files, not to a file
    data_dir = Path(r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\Weather_Energy")
    
    # Path to the specific file
    specific_file = data_dir / "weather_energy.csv"
    
    if os.path.exists(specific_file):
        print(f"Processing file: {specific_file}")
        
        # Load data
        df = load_data(specific_file)
        
        # Aggregate to 15-minute intervals
        df_15min = aggregate_to_15min(df)
        
        # Create output filename
        output_file = str(specific_file).replace(".csv", "_15min.csv")
        
        # Save aggregated data
        save_aggregated_data(df_15min, output_file)
        print(f"Completed processing file: {specific_file}")
    else:
        # Find input files with 5min in the name
        input_files = list(data_dir.glob("*5min*.csv"))
        
        if not input_files:
            print("No 5-minute granularity files found!")
            return
        
        for input_file in input_files:
            print(f"Processing file: {input_file}")
            
            # Load data
            df = load_data(input_file)
            
            # Aggregate to 15-minute intervals
            df_15min = aggregate_to_15min(df)
            
            # Create output filename
            output_file = str(input_file).replace("5min", "15min")
            
            # Save aggregated data
            save_aggregated_data(df_15min, output_file)
            print(f"Completed processing file: {input_file}")

if __name__ == "__main__":
    main()