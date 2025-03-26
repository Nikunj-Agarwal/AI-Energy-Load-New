import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from tqdm import tqdm

# Import from config file
from config import (
    PATHS, THEME_CATEGORIES, setup_and_verify, test_directory_writing as config_test_directory_writing
)

#==============================
# CONFIGURATION VARIABLES
#==============================

# File paths from config
INPUT_FILE_PATH = os.path.join(PATHS["RAW_DATA_DIR"], "delhi_gkg_data_2021_jan1_3.csv")
OUTPUT_FILE_PATH = os.path.join(PATHS["PROCESSED_DIR"], "processed_gkg_parsed_data.csv")

# Batch processing settings
BATCH_DIR = PATHS["BATCH_DIR"]
MERGED_OUTPUT_PATH = os.path.join(PATHS["PROCESSED_DIR"], "merged_delhi_gkg_data.csv")
PROCESS_BATCHES = True  # Set to True to process batch files instead of single file

# Processing options
SAMPLE_SIZE = None  # Set to None for full dataset processing

# Theme categories and keywords
theme_categories = THEME_CATEGORIES

# Tone metrics to extract
TONE_METRICS = ['tone', 'positive', 'negative', 'polarity', 'activity', 'self_ref']

#==============================
# DATA VALIDATION FUNCTIONS - NEW!
#==============================

def validate_input_data(df, required_columns=None):
    """Validate that input data has the required columns and structure"""
    if required_columns is None:
        required_columns = ['DATE', 'V2Themes', 'V2Tone', 'Counts', 'Amounts']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"ERROR: Input data missing required columns: {missing_columns}")
        return False
    
    return True

def validate_output_data(df):
    """Verify the processed data has all fields required by aggregation script"""
    expected_theme_columns = [f'theme_{category}' for category in theme_categories]
    expected_tone_columns = [f'tone_{metric}' for metric in TONE_METRICS]
    
    required_columns = ['datetime', 'GKGRECORDID'] + expected_theme_columns + expected_tone_columns
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"WARNING: Processed data missing expected columns: {missing_columns}")
        print("This may cause issues in downstream aggregation.")
        return False
    
    return True

def save_column_headers(df, output_dir):
    """Save all column headers to a reference file for debugging and documentation"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    headers_file = os.path.join(output_dir, "column_headers_reference.txt")
    
    with open(headers_file, 'w', encoding='utf-8') as f:
        f.write(f"# Column Headers Reference - Generated on {datetime.now()}\n")
        f.write(f"# Total columns: {len(df.columns)}\n\n")
        
        # Group columns by prefix for better organization
        prefixes = {
            'theme_': 'Theme columns',
            'tone_': 'Tone metrics',
            'entity_': 'Entity metrics',
            'parsed_': 'Parsed data columns'
        }
        
        # First write core columns
        f.write("## Core columns\n")
        core_cols = [col for col in df.columns if not any(col.startswith(p) for p in prefixes)]
        for col in core_cols:
            f.write(f"- {col}\n")
        
        # Write grouped columns
        for prefix, title in prefixes.items():
            cols = [col for col in df.columns if col.startswith(prefix)]
            if cols:
                f.write(f"\n## {title}\n")
                for col in cols:
                    f.write(f"- {col}\n")
    
    print(f"Column headers saved to {headers_file}")
    return headers_file

def ensure_theme_category_consistency():
    """Ensure theme categories are properly capitalized and consistent"""
    global theme_categories
    
    # Make a copy with properly capitalized keys
    updated_categories = {}
    for category, keywords in theme_categories.items():
        # Capitalize first letter of category name for consistency
        updated_key = category.capitalize()
        updated_categories[updated_key] = keywords
    
    theme_categories = updated_categories
    return theme_categories

#==============================
# HELPER FUNCTIONS
#==============================

def convert_date(date_str):
    """Convert GDELT date format to datetime"""
    try:
        date_str = str(date_str).zfill(14)  # Ensure 14 digits
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(date_str[8:10])
        minute = int(date_str[10:12])
        second = int(date_str[12:14])
        return datetime(year, month, day, hour, minute, second)
    except:
        return None

def parse_themes(theme_str):
    """Extract themes from V2Themes column"""
    if pd.isna(theme_str):
        return {}
    
    themes_dict = {}
    themes = theme_str.split(';')
    
    for theme in themes:
        if not theme:
            continue
        
        parts = theme.split(',')
        if len(parts) >= 1:
            theme_name = parts[0].lower()
            themes_dict[theme_name] = 1
    
    return themes_dict

def categorize_themes(themes_dict):
    """Map individual themes to theme categories"""
    categorized = {category: 0 for category in theme_categories}
    
    for theme in themes_dict:
        for category, keywords in theme_categories.items():
            if any(keyword.lower() in theme.lower() for keyword in keywords):
                categorized[category] = 1
                break
    
    return categorized

def parse_tone(tone_str):
    """Extract tone metrics from V2Tone column"""
    if pd.isna(tone_str):
        return {metric: 0 for metric in TONE_METRICS}
    
    values = tone_str.split(',')
    
    if len(values) < 6:
        return {metric: 0 for metric in TONE_METRICS}
    
    try:
        return {
            'tone': float(values[0]),
            'positive': float(values[1]),
            'negative': float(values[2]),
            'polarity': float(values[3]),
            'activity': float(values[4]),
            'self_ref': float(values[5])
        }
    except ValueError:
        # Handle conversion errors
        return {metric: 0 for metric in TONE_METRICS}

def parse_counts(counts_str):
    """Extract entity counts"""
    if pd.isna(counts_str):
        return {}
    
    counts_dict = {}
    counts = counts_str.split(';')
    
    for count in counts:
        if not count:
            continue
        
        parts = count.split('#')
        if len(parts) >= 2:
            entity = parts[0].lower()
            try:
                count_value = int(parts[1])
                counts_dict[entity] = count_value
            except:
                pass
    
    return counts_dict

def parse_amounts(amounts_str):
    """Extract numerical amounts mentioned in text"""
    if pd.isna(amounts_str):
        return []
    
    amounts = []
    amount_entries = amounts_str.split(';')
    
    for entry in amount_entries:
        if not entry:
            continue
        
        try:
            amount_value = float(entry.split(',')[0])
            amounts.append(amount_value)
        except:
            pass
    
    return amounts

# Add after the parse_amounts function

def calculate_energy_impact_score(row):
    """Calculate a composite impact score for energy load prediction"""
    # Base impact score starts at 1
    impact = 1.0
    
    # If amounts exist, use them as a more reliable proxy than counts
    if 'parsed_amounts' in row and row['parsed_amounts'] and len(row['parsed_amounts']) > 0:
        # Log scale to prevent extreme outliers from dominating
        impact *= (1 + np.log1p(min(np.mean(row['parsed_amounts']), 10000)))
    # Fallback to entity count if available
    elif 'entity_count' in row and row['entity_count'] > 0:
        impact *= (1 + np.log1p(min(row['entity_count'], 1000)))
    
    # Energy-specific amplification
    if 'theme_Energy' in row and row['theme_Energy'] > 0:
        impact *= 2.0
    
    # Infrastructure affects energy systems
    if 'theme_Infrastructure' in row and row['theme_Infrastructure'] > 0:
        impact *= 1.5
    
    # Environmental themes often correlate with energy consumption changes
    if 'theme_Environment' in row and row['theme_Environment'] > 0:
        impact *= 1.3
    
    # Tone amplification - negative news about energy systems has bigger impact
    if 'tone_negative' in row and row['theme_Energy'] > 0 and row['tone_negative'] > 0:
        impact *= (1 + row['tone_negative'] * 0.5)
    
    # Cap at a reasonable maximum to prevent extreme outliers
    return min(impact, 100.0)

# Add a new function to extract contextual energy information
def extract_energy_contexts(themes_dict, text_field=None):
    """Extract energy-specific contexts from themes and text"""
    # Initialize energy context tags
    contexts = {
        'energy_supply': 0,
        'energy_demand': 0,
        'energy_price': 0,
        'energy_policy': 0,
        'energy_infrastructure': 0,
        'renewable_energy': 0,
        'fossil_fuel': 0,
        'weather_event': 0
    }
    
    # Energy supply keywords
    supply_keywords = ['production', 'output', 'generation', 'supply']
    
    # Energy demand keywords
    demand_keywords = ['consumption', 'demand', 'usage', 'load']
    
    # Price keywords
    price_keywords = ['price', 'cost', 'tariff', 'rate', 'economic']
    
    # Policy keywords
    policy_keywords = ['policy', 'regulation', 'law', 'government', 'subsidy']
    
    # Infrastructure keywords
    infra_keywords = ['grid', 'blackout', 'outage', 'infrastructure', 'plant', 'station']
    
    # Renewable keywords
    renewable_keywords = ['solar', 'wind', 'hydro', 'renewable', 'clean energy', 'green energy']
    
    # Fossil fuel keywords
    fossil_keywords = ['oil', 'gas', 'coal', 'petroleum', 'diesel', 'gasoline']
    
    # Weather event keywords
    weather_keywords = ['storm', 'hurricane', 'temperature', 'heat', 'cold', 'snow', 'rain']
    
    # Check presence in themes
    for theme in themes_dict:
        theme_lower = theme.lower()
        
        # Check each context category
        for keyword in supply_keywords:
            if keyword in theme_lower:
                contexts['energy_supply'] = 1
                
        for keyword in demand_keywords:
            if keyword in theme_lower:
                contexts['energy_demand'] = 1
                
        # Continue with checks for other categories
        # ... similar checks for other categories
    
    # Optional: Check in text field if provided
    if text_field and not pd.isna(text_field):
        text_lower = text_field.lower()
        # Perform similar keyword checks in the text
        # ... checks similar to above
    
    return contexts

#==============================
# BATCH PROCESSING FUNCTIONS
#==============================

def merge_batch_files(batch_dir=BATCH_DIR, output_path=MERGED_OUTPUT_PATH):
    """Merge all batch CSV files into a single file"""
    print(f"Looking for batch files in {batch_dir}...")
    
    # Find all CSV files in the batch directory
    batch_files = glob.glob(os.path.join(batch_dir, "*.csv"))
    
    if not batch_files:
        print("Error: No batch files found!")
        return None
    
    print(f"Found {len(batch_files)} batch files. Merging...")
    
    # Read and concatenate all batch files
    dfs = []
    total_rows = 0
    
    for file_path in tqdm(batch_files, desc="Reading batch files"):
        try:
            # Add encoding parameter
            df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
            dfs.append(df)
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            try:
                # Fallback to latin-1 encoding
                print(f"Retrying with latin-1 encoding...")
                df = pd.read_csv(file_path, encoding='latin-1')
                dfs.append(df)
                total_rows += len(df)
            except Exception as e2:
                print(f"Failed with latin-1 encoding too: {e2}")
    
    if not dfs:
        print("No valid data found in batch files!")
        return None
    
    # Concatenate all dataframes
    try:
        merged_df = pd.concat(dfs, ignore_index=True)
    except ValueError as e:
        print(f"Error merging batch files: {e}")
        return None
    
    print(f"Merged {len(batch_files)} files with a total of {total_rows} rows.")
    
    # Add encoding parameter to the save operation
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Merged data saved to {output_path}")
    
    return merged_df

#==============================
# ENHANCED PROCESS DATAFRAME FUNCTION
#==============================

from functools import lru_cache

# Add caching to expensive functions
@lru_cache(maxsize=10000)
def parse_themes_cached(theme_str):
    """Cached version of parse_themes for better performance"""
    return parse_themes(theme_str)

# Modify the process_dataframe function to include article_count
def process_dataframe(df):
    """Process a dataframe with all the parsing functions"""
    print(f"Processing dataframe with {len(df)} rows...")
    
    # Validate input data first - NEW!
    if not validate_input_data(df):
        print("ERROR: Input data validation failed. Processing may produce incomplete results.")
    
    # Sample a subset for testing if requested
    if SAMPLE_SIZE:
        sample_size = min(SAMPLE_SIZE, len(df))
        df = df.head(sample_size)
        print(f"Using sample of {sample_size} rows")
    
    # Convert DATE to datetime
    print("Converting dates...")
    df['datetime'] = df['DATE'].apply(convert_date)
    
    # Handle invalid dates - NEW!
    invalid_dates = df['datetime'].isna().sum()
    if (invalid_dates > 0):
        print(f"WARNING: {invalid_dates} rows with invalid dates. These will be excluded.")
        df = df.dropna(subset=['datetime'])
    
    # Ensure consistent theme categories - NEW!
    ensure_theme_category_consistency()
    
    # Process themes with caching
    print("Processing themes with caching...")
    df['parsed_themes'] = df['V2Themes'].apply(lambda x: parse_themes_cached(str(x)) if pd.notna(x) else {})
    df['theme_categories'] = df['parsed_themes'].apply(categorize_themes)
    
    # Create columns for each theme category
    for category in theme_categories:
        df[f'theme_{category}'] = df['theme_categories'].apply(
            lambda x: x.get(category, 0) if isinstance(x, dict) else 0
        )
    
    # Process tone metrics
    print("Processing tone metrics...")
    df['parsed_tone'] = df['V2Tone'].apply(parse_tone)
    for metric in TONE_METRICS:
        df[f'tone_{metric}'] = df['parsed_tone'].apply(lambda x: x.get(metric, 0))
    
    # Process counts and amounts
    print("Processing counts and amounts...")
    df['parsed_counts'] = df['Counts'].apply(parse_counts)
    df['entity_count'] = df['parsed_counts'].apply(lambda x: sum(x.values()) if isinstance(x, dict) else 0)
    df['entity_variety'] = df['parsed_counts'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    
    df['parsed_amounts'] = df['Amounts'].apply(parse_amounts)
    df['avg_amount'] = df['parsed_amounts'].apply(lambda x: np.mean(x) if x else 0)
    df['max_amount'] = df['parsed_amounts'].apply(lambda x: max(x) if x else 0)
    df['amount_count'] = df['parsed_amounts'].apply(lambda x: len(x) if x else 0)
    
    # Add article_count column (1 per row) to match aggregation expectations
    df['article_count'] = 1
    
    # Add enhanced energy impact score
    print("Calculating energy impact scores...")
    df['energy_impact_score'] = df.apply(calculate_energy_impact_score, axis=1)
    
    # Extract energy contexts if V2FullNames or TEXT fields are available
    text_field = None
    if 'V2FullNames' in df.columns:
        text_field = 'V2FullNames'
    elif 'TEXT' in df.columns:
        text_field = 'TEXT'
    
    if text_field:
        print(f"Extracting energy contexts from {text_field}...")
        energy_contexts = df.apply(
            lambda row: extract_energy_contexts(row['parsed_themes'], row[text_field]), 
            axis=1
        )
        
        # Add each context as a column
        for context in ['energy_supply', 'energy_demand', 'energy_price', 
                       'energy_policy', 'energy_infrastructure', 
                       'renewable_energy', 'fossil_fuel', 'weather_event']:
            df[context] = energy_contexts.apply(lambda x: x[context])
    
    # Create amount-theme interaction features
    if 'avg_amount' in df.columns:
        df['energy_amount'] = df['theme_Energy'] * df['avg_amount']
        df['infrastructure_amount'] = df['theme_Infrastructure'] * df['avg_amount']
        df['environment_amount'] = df['theme_Environment'] * df['avg_amount']
    
    # Add article_count column (1 per row) to match aggregation expectations
    df['article_count'] = 1
    
    # Validate the output data - NEW!
    validate_output_data(df)
    
    # Print some statistics about the parsed data
    print("\nTheme distribution in data:")
    for category in theme_categories:
        count = df[f'theme_{category}'].sum()
        print(f"{category}: {count}")
    
    print("\nTone statistics:")
    print(f"Average tone: {df['tone_tone'].mean()}")
    print(f"Most positive: {df['tone_positive'].max()}")
    print(f"Most negative: {df['tone_negative'].max()}")
    
    return df

#==============================
# IMPROVED MAIN PROCESSING FUNCTIONS
#==============================

def test_directory_access():
    """Test if we can write to all required directories"""
    print("Testing directory access and file writing permissions...")
    
    test_dirs = [
        os.path.dirname(INPUT_FILE_PATH),  # ADDED input directory
        os.path.dirname(OUTPUT_FILE_PATH),
        os.path.dirname(MERGED_OUTPUT_PATH),
        BATCH_DIR
    ]
    
    all_passed = True
    for directory in test_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"ERROR: Could not create directory {directory}: {e}")
                all_passed = False
                continue
        
        test_file = os.path.join(directory, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write(f"Test write at {datetime.now()}")
            os.remove(test_file)
            print(f"✓ Successfully wrote to {directory}")
        except Exception as e:
            print(f"✗ ERROR: Could not write to {directory}: {e}")
            all_passed = False
    
    # ADDED - Test DataFrame writing to output directories
    test_df = pd.DataFrame({'test': [1, 2, 3]})
    
    # Test output directory
    try:
        test_csv = os.path.join(os.path.dirname(OUTPUT_FILE_PATH), "test_dataframe.csv")
        test_df.to_csv(test_csv, index=False)
        os.remove(test_csv)
        print(f"✓ Successfully wrote DataFrame to {os.path.dirname(OUTPUT_FILE_PATH)}")
    except Exception as e:
        print(f"✗ ERROR: Could not write DataFrame to {os.path.dirname(OUTPUT_FILE_PATH)}: {e}")
        all_passed = False
    
    # Test merged output directory
    try:
        test_csv = os.path.join(os.path.dirname(MERGED_OUTPUT_PATH), "test_dataframe.csv")
        test_df.to_csv(test_csv, index=False)
        os.remove(test_csv)
        print(f"✓ Successfully wrote DataFrame to {os.path.dirname(MERGED_OUTPUT_PATH)}")
    except Exception as e:
        print(f"✗ ERROR: Could not write DataFrame to {os.path.dirname(MERGED_OUTPUT_PATH)}: {e}")
        all_passed = False
    
    return all_passed

def process_single_file(input_path=INPUT_FILE_PATH, output_path=OUTPUT_FILE_PATH):
    """Process a single GKG data file"""
    # Verify directory structure using config function
    if not setup_and_verify():
        print("ERROR: Directory setup verification failed. Aborting processing.")
        return
    
    # Test directory access
    if not test_directory_access():
        print("ERROR: Directory access test failed. Aborting processing.")
        return
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data with encoding handling
    print(f"Loading data from {input_path}...")
    try:
        # First try with utf-8 and error replacement
        df = pd.read_csv(input_path, encoding='utf-8', encoding_errors='replace')
        print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error with utf-8 encoding, trying latin-1: {e}")
        try:
            # Fall back to latin-1 which can handle any byte sequence
            df = pd.read_csv(input_path, encoding='latin-1')
            print(f"Data loaded with latin-1 encoding: {df.shape[0]} rows")
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    
    # Process the dataframe
    processed_df = process_dataframe(df)
    
    # Verify we have data before saving - NEW!
    if processed_df is None or len(processed_df) == 0:
        print("ERROR: Processing resulted in empty dataframe. No output will be saved.")
        return
    
    # Save the processed data with improved error handling - NEW!
    try:
        processed_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nProcessed data saved to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save output to {output_path}: {e}")

def process_batch_files():
    """Merge and process all batch files"""
    # Verify directory structure using config function
    if not setup_and_verify():
        print("ERROR: Directory setup verification failed. Aborting processing.")
        return
    
    # Test directory access
    if not test_directory_access():
        print("ERROR: Directory access test failed. Aborting batch processing.")
        return
    
    # First merge all batch files
    print(f"Merging batch files from {BATCH_DIR}...")
    merged_df = merge_batch_files()
    
    if merged_df is None:
        print("ERROR: Batch file merging failed. No data to process.")
        return
        
    if len(merged_df) == 0:
        print("ERROR: Merged data is empty. No data to process.")
        return
    
    print(f"Successfully merged batch files. Processing {len(merged_df)} rows...")
    
    # Process the merged dataframe
    processed_df = process_dataframe(merged_df)
    
    # Verify we have data before saving
    if processed_df is None or len(processed_df) == 0:
        print("ERROR: Processing resulted in empty dataframe. No output will be saved.")
        return
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed data with improved error handling
    try:
        processed_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8')
        print(f"\nProcessed merged data saved to {OUTPUT_FILE_PATH}")
        
        # Output some statistics about the processed data
        print(f"Processed {len(processed_df)} rows with {processed_df.shape[1]} columns")
        theme_columns = [col for col in processed_df.columns if col.startswith('theme_')]
        print(f"Extracted {len(theme_columns)} theme categories")
        tone_columns = [col for col in processed_df.columns if col.startswith('tone_')]
        print(f"Extracted {len(tone_columns)} tone metrics")
    except Exception as e:
        print(f"ERROR: Failed to save output to {OUTPUT_FILE_PATH}: {e}")

def process_batch_files_in_chunks():
    """Process batch files with memory-efficient chunking"""
    # Verify directories first
    if not setup_and_verify() or not test_directory_access():
        return
    
    # Find all batch files
    batch_files = glob.glob(os.path.join(BATCH_DIR, "*.csv"))
    if not batch_files:
        print("Error: No batch files found!")
        return
    
    print(f"Found {len(batch_files)} batch files")
    
    # Create output directory
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # First chunk needs to create the file, others will append
    first_chunk = True
    # Flag to save column headers once
    headers_saved = False
    
    for file_idx, file_path in enumerate(batch_files):
        print(f"\nProcessing batch file {file_idx+1}/{len(batch_files)}: {os.path.basename(file_path)}")
        
        # Process in chunks to save memory
        chunk_size = 50000  # Adjust based on your system's memory
        
        try:
            for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8')):
                print(f"  Processing chunk {chunk_idx+1} ({len(chunk)} rows)")
                
                # Process this chunk
                processed_chunk = process_dataframe(chunk)
                
                if processed_chunk is not None and len(processed_chunk) > 0:
                    # Save column headers to a reference file (only once)
                    if not headers_saved:
                        save_column_headers(processed_chunk, output_dir)
                        headers_saved = True
                        
                    # Write directly to file with UTF-8 encoding
                    if first_chunk:
                        # First chunk creates the file with headers
                        processed_chunk.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8')
                        first_chunk = False
                    else:
                        # Subsequent chunks append without headers
                        processed_chunk.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, 
                                              index=False, encoding='utf-8')
                    
                    print(f"    Wrote {len(processed_chunk)} rows to output file")
                        
                # Clear memory
                del processed_chunk
                del chunk
        except Exception as e:
            try:
                print(f"  Error with UTF-8 encoding, trying latin-1: {e}")
                for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, encoding='latin-1')):
                    print(f"  Processing chunk {chunk_idx+1} ({len(chunk)} rows) with latin-1 encoding")
                    
                    # Process this chunk
                    processed_chunk = process_dataframe(chunk)
                    
                    if processed_chunk is not None and len(processed_chunk) > 0:
                        # Save column headers to a reference file (only once)
                        if not headers_saved:
                            save_column_headers(processed_chunk, output_dir)
                            headers_saved = True
                            
                        # Write directly to file with UTF-8 encoding
                        if first_chunk:
                            # First chunk creates the file with headers
                            processed_chunk.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8')
                            first_chunk = False
                        else:
                            # Subsequent chunks append without headers
                            processed_chunk.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, 
                                                  index=False, encoding='utf-8')
                        
                        print(f"    Wrote {len(processed_chunk)} rows to output file")
                            
                    # Clear memory
                    del processed_chunk
                    del chunk
            except Exception as e2:
                print(f"  Error processing file {os.path.basename(file_path)}: {e2}")
                print("  Continuing with next file...")
    
    print(f"Processing complete. Output saved to {OUTPUT_FILE_PATH}")
    
    # Generate both reports after all processing is done
    try:
        print("Generating data quality reports...")
        
        # Basic report with sample (quick)
        print("1. Generating sample-based quality report...")
        sample_df = pd.read_csv(OUTPUT_FILE_PATH, nrows=100000, encoding='utf-8', encoding_errors='replace')
        log_data_quality_metrics(sample_df, output_dir)
        print("Sample-based quality report generated successfully")
        
        # Comprehensive report across all data (memory-efficient)
        print("2. Generating comprehensive quality report across all data...")
        generate_comprehensive_quality_report(OUTPUT_FILE_PATH, output_dir, chunk_size=50000)
        print("Comprehensive data quality report generated successfully")
        
    except Exception as e:
        print(f"Error generating data quality reports: {e}")

# Add to gkg_data_sparsing.py
def log_data_quality_metrics(df, output_dir):
    """Log data quality metrics to help identify issues"""
    log_path = os.path.join(output_dir, "data_quality_log.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"GDELT Data Quality Report - {datetime.now()}\n")
        f.write("="*80 + "\n\n")
        
        # Date range
        f.write(f"Date Range: {df['datetime'].min()} to {df['datetime'].max()}\n\n")
        
        # Record count
        f.write(f"Total Records: {len(df)}\n")
        
        # Missing values
        null_counts = df.isnull().sum()
        f.write("\nMissing Values:\n")
        for col, count in null_counts.items():
            if count > 0:
                f.write(f"  {col}: {count} ({count/len(df)*100:.2f}%)\n")
        
        # Theme distribution
        f.write("\nTheme Distribution:\n")
        for category in theme_categories:
            count = df[f'theme_{category}'].sum()
            f.write(f"  {category}: {count} ({count/len(df)*100:.2f}%)\n")
        
        # Tone statistics
        f.write("\nTone Statistics:\n")
        f.write(f"  Average Tone: {df['tone_tone'].mean():.4f}\n")
        f.write(f"  Tone Range: {df['tone_tone'].min():.4f} to {df['tone_tone'].max():.4f}\n")
        f.write(f"  Most Positive: {df['tone_positive'].max():.4f}\n")
        f.write(f"  Most Negative: {df['tone_negative'].max():.4f}\n")
        
        # Potential anomalies
        extreme_tone = len(df[df['tone_tone'].abs() > 10])
        if extreme_tone > 0:
            f.write(f"\nPotential Anomalies:\n")
            f.write(f"  Records with extreme tone (>10): {extreme_tone}\n")
        
        no_themes = len(df[(df[[f'theme_{c}' for c in theme_categories]].sum(axis=1) == 0)])
        if no_themes > 0:
            f.write(f"  Records with no themes detected: {no_themes}\n")
    
    print(f"Data quality report saved to {log_path}")
    return log_path

# Add to gkg_data_sparsing.py
def process_incremental():
    """Process only new batch files that haven't been processed yet"""
    if not setup_and_verify() or not test_directory_access():
        return
    
    # Get list of batch files
    batch_files = glob.glob(os.path.join(BATCH_DIR, "*.csv"))
    if not batch_files:
        print("No batch files found.")
        return
    
    # Check if output file exists
    if os.path.exists(OUTPUT_FILE_PATH):
        # Load existing processed data with encoding
        try:
            existing_df = pd.read_csv(OUTPUT_FILE_PATH, encoding='utf-8', encoding_errors='replace')
            existing_ids = set(existing_df['GKGRECORDID']) if 'GKGRECORDID' in existing_df.columns else set()
            print(f"Found existing output with {len(existing_ids)} records")
        except Exception as e:
            print(f"Error reading existing output file: {e}")
            try:
                # Try with latin-1 encoding
                existing_df = pd.read_csv(OUTPUT_FILE_PATH, encoding='latin-1')
                existing_ids = set(existing_df['GKGRECORDID']) if 'GKGRECORDID' in existing_df.columns else set()
                print(f"Found existing output with {len(existing_ids)} records (using latin-1 encoding)")
            except:
                print("Error reading existing output file. Will process all files.")
                existing_ids = set()
    else:
        existing_ids = set()
    
    # Process new files incrementally
    for file_idx, file_path in enumerate(batch_files):
        filename = os.path.basename(file_path)
        print(f"Checking file {file_idx+1}/{len(batch_files)}: {filename}")
        
        # Read file with encoding
        try:
            batch_df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
            # Filter to only new records
            if 'GKGRECORDID' in batch_df.columns:
                new_records = batch_df[~batch_df['GKGRECORDID'].isin(existing_ids)]
                if len(new_records) == 0:
                    print(f"  No new records in {filename}, skipping")
                    continue
                print(f"  Found {len(new_records)} new records in {filename}")
                
                # Process new records
                processed_new = process_dataframe(new_records)
                
                # Append to existing output with encoding
                if processed_new is not None and len(processed_new) > 0:
                    processed_new.to_csv(OUTPUT_FILE_PATH, mode='a', 
                                         header=not os.path.exists(OUTPUT_FILE_PATH), 
                                         index=False, encoding='utf-8')
                    print(f"  Appended {len(processed_new)} processed records to output")
                    
                    # Update existing IDs
                    existing_ids.update(processed_new['GKGRECORDID'])
            else:
                print(f"  Warning: File {filename} doesn't have GKGRECORDID column")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    print("Incremental processing complete")

def generate_comprehensive_quality_report(file_path, output_dir, chunk_size=50000):
    """Generate a comprehensive data quality report across all data without loading everything at once"""
    log_path = os.path.join(output_dir, "comprehensive_data_quality_report.txt")
    print(f"Generating comprehensive data quality report for {file_path}")
    
    # Initialize counters and statistics
    total_records = 0
    missing_values = {}
    theme_counts = {category: 0 for category in theme_categories}
    tone_stats = {
        'tone_sum': 0, 
        'positive_max': float('-inf'),
        'negative_max': float('-inf'),
        'tone_min': float('inf'),
        'tone_max': float('-inf')
    }
    
    # Track min and max dates - initialize properly
    min_date = pd.Timestamp.max  # Start with maximum possible date
    max_date = pd.Timestamp.min  # Start with minimum possible date
    valid_dates_found = False
    
    # Date distribution by month (YYYY-MM format)
    monthly_counts = {}
    # Also track yearly distribution
    yearly_counts = {}
    
    # Track anomalies
    extreme_tone_count = 0
    no_themes_count = 0
    
    # Process the file in chunks
    print("Scanning data in chunks to gather statistics...")
    try:
        # Get total number of chunks for better progress reporting
        total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='replace'))
        total_chunks = total_rows // chunk_size + 1
        print(f"File has approximately {total_rows} rows, will process in {total_chunks} chunks")
        
        # Use tqdm for a progress bar if possible
        chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8', encoding_errors='replace')
        
        for i, chunk in enumerate(tqdm(chunk_reader, desc="Processing chunks", total=total_chunks)):
            # Update total records
            current_chunk_size = len(chunk)
            total_records += current_chunk_size
            
            # Track min and max dates with debug output
            if 'datetime' in chunk.columns:
                # First convert to datetime with proper error handling
                chunk['datetime'] = pd.to_datetime(chunk['datetime'], errors='coerce')
                
                # Filter out NaT values
                valid_dates = chunk['datetime'].dropna()
                
                if not valid_dates.empty:
                    chunk_min = valid_dates.min()
                    chunk_max = valid_dates.max()
                    
                    # Update global min/max dates
                    if chunk_min < min_date:
                        min_date = chunk_min
                        print(f"New min date found: {min_date}")
                    
                    if chunk_max > max_date:
                        max_date = chunk_max
                        print(f"New max date found: {max_date}")
                    
                    valid_dates_found = True
                    
                    # Count records by month and year
                    month_groups = valid_dates.dt.strftime('%Y-%m').value_counts()
                    for month, count in month_groups.items():
                        monthly_counts[month] = monthly_counts.get(month, 0) + count
                    
                    year_groups = valid_dates.dt.strftime('%Y').value_counts()
                    for year, count in year_groups.items():
                        yearly_counts[year] = yearly_counts.get(year, 0) + count
            
            # Track missing values
            chunk_nulls = chunk.isnull().sum()
            for col, count in chunk_nulls.items():
                if count > 0:
                    missing_values[col] = missing_values.get(col, 0) + count
            
            # Track theme distribution
            for category in theme_categories:
                col_name = f'theme_{category}'
                if col_name in chunk.columns:
                    theme_counts[category] += chunk[col_name].sum()
            
            # Track tone statistics
            if 'tone_tone' in chunk.columns:
                tone_stats['tone_sum'] += chunk['tone_tone'].sum()
                
                chunk_tone_max = chunk['tone_tone'].max()
                if not pd.isna(chunk_tone_max) and chunk_tone_max > tone_stats['tone_max']:
                    tone_stats['tone_max'] = chunk_tone_max
                
                chunk_tone_min = chunk['tone_tone'].min()
                if not pd.isna(chunk_tone_min) and chunk_tone_min < tone_stats['tone_min']:
                    tone_stats['tone_min'] = chunk_tone_min
            
            if 'tone_positive' in chunk.columns:
                chunk_positive_max = chunk['tone_positive'].max()
                if not pd.isna(chunk_positive_max) and chunk_positive_max > tone_stats['positive_max']:
                    tone_stats['positive_max'] = chunk_positive_max
            
            if 'tone_negative' in chunk.columns:
                chunk_negative_max = chunk['tone_negative'].max()
                if not pd.isna(chunk_negative_max) and chunk_negative_max > tone_stats['negative_max']:
                    tone_stats['negative_max'] = chunk_negative_max
            
            # Track anomalies
            if 'tone_tone' in chunk.columns:
                extreme_tone_count += len(chunk[chunk['tone_tone'].abs() > 10])
            
            # Check for records with no themes - FIX THIS PART
            theme_cols = [col for col in chunk.columns if col.startswith('theme_')]
            if theme_cols:
                # Make sure we're only working with numeric columns
                numeric_theme_cols = []
                for col in theme_cols:
                    if pd.api.types.is_numeric_dtype(chunk[col]):
                        numeric_theme_cols.append(col)
                    else:
                        # Try to convert non-numeric columns to numeric
                        try:
                            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                            numeric_theme_cols.append(col)
                        except:
                            print(f"Warning: Skipping non-numeric theme column: {col}")
                
                # Now use only the numeric columns for calculating no themes
                if numeric_theme_cols:
                    no_themes_in_chunk = len(chunk[chunk[numeric_theme_cols].sum(axis=1) == 0])
                    no_themes_count += no_themes_in_chunk
                else:
                    print("Warning: No numeric theme columns found to calculate no-theme records")
            
            # Show progress regularly
            if (i+1) % 10 == 0 or i == 0:  # First chunk and then every 10th chunk
                print(f"Processed {total_records:,} records ({(i+1)/total_chunks*100:.1f}% complete)")
                print(f"Current date range: {min_date} to {max_date}")
        
        # Adjust dates if none found
        if not valid_dates_found:
            min_date = pd.NaT
            max_date = pd.NaT
        
        # Calculate average tone
        avg_tone = tone_stats['tone_sum'] / total_records if total_records > 0 else 0
        
        # Write the report
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"COMPREHENSIVE GDELT Data Quality Report - {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            
            # Date range with clear marking
            f.write(f"DATE RANGE SUMMARY\n")
            f.write(f"==================\n")
            f.write(f"Full Date Range: {min_date} to {max_date}\n")
            f.write(f"Total Timespan: {(max_date - min_date).days} days\n\n")
            
            # Years covered
            f.write(f"Years Covered: {', '.join(sorted(yearly_counts.keys()))}\n\n")
            
            # Record count
            f.write(f"Total Records: {total_records:,}\n\n")
            
            # Distribution by year (new)
            f.write("Records by Year:\n")
            for year, count in sorted(yearly_counts.items()):
                f.write(f"  {year}: {count:,} ({count/total_records*100:.2f}%)\n")
            f.write("\n")
            
            # Distribution by month
            f.write("Records by Month (YYYY-MM):\n")
            for month, count in sorted(monthly_counts.items()):
                f.write(f"  {month}: {count:,} ({count/total_records*100:.2f}%)\n")
            f.write("\n")
            
            # Missing values
            f.write("Missing Values:\n")
            for col, count in missing_values.items():
                f.write(f"  {col}: {count:,} ({count/total_records*100:.2f}%)\n")
            f.write("\n")
            
            # Theme distribution
            f.write("Theme Distribution:\n")
            for category, count in theme_counts.items():
                f.write(f"  {category}: {count:,} ({count/total_records*100:.2f}%)\n")
            f.write("\n")
            
            # Tone statistics
            f.write("Tone Statistics:\n")
            f.write(f"  Average Tone: {avg_tone:.4f}\n")
            f.write(f"  Tone Range: {tone_stats['tone_min']:.4f} to {tone_stats['tone_max']:.4f}\n")
            f.write(f"  Most Positive: {tone_stats['positive_max']:.4f}\n")
            f.write(f"  Most Negative: {tone_stats['negative_max']:.4f}\n")
            f.write("\n")
            
            # Potential anomalies
            f.write("Potential Anomalies:\n")
            f.write(f"  Records with extreme tone (>10): {extreme_tone_count:,} ({extreme_tone_count/total_records*100:.2f}%)\n")
            f.write(f"  Records with no themes detected: {no_themes_count:,} ({no_themes_count/total_records*100:.2f}%)\n")
        
        print(f"Comprehensive data quality report saved to {log_path}")
        print(f"Full date range: {min_date} to {max_date} ({(max_date - min_date).days} days)")
        return log_path
    
    except Exception as e:
        print(f"Error generating comprehensive data quality report: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT GKG Data Sparsing")
    parser.add_argument("--mode", choices=["single", "batch", "incremental", "chunks"], 
                        default="chunks", help="Processing mode")
    parser.add_argument("--input", help="Input file path (for single mode)")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--sample", type=int, help="Sample size for testing")
    
    args = parser.parse_args()
    
    print(f"GDELT GKG Data Sparsing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Override global settings with command line args
    if args.input:
        INPUT_FILE_PATH = args.input
    if args.output:
        OUTPUT_FILE_PATH = args.output
    if args.sample:
        SAMPLE_SIZE = args.sample
    
    # Choose processing mode
    if args.mode == "single":
        print(f"Mode: Processing single file: {INPUT_FILE_PATH}")
        process_single_file()
    elif args.mode == "batch":
        print("Mode: Processing all batch files")
        process_batch_files()
    elif args.mode == "incremental":
        print("Mode: Incremental processing of new batch files")
        process_incremental()
    elif args.mode == "chunks":
        print("Mode: Processing batch files in memory-efficient chunks")
        process_batch_files_in_chunks()
    
    print(f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")