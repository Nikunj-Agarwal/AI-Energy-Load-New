"""
GDELT Project Configuration
---------------------------
Central configuration file for all GDELT scripts.
This file:
1. Defines all paths and directories used across the project
2. Sets up the proper directory structure
3. Validates write access to all directories
4. Provides utility functions for path management
5. Defines shared constants like theme categories
"""
import os
import sys
from datetime import datetime

# ===== BASE PATHS =====
# Base directory for all outputs
OUTPUT_DIR = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\OUTPUT_DIR"

# ===== DIRECTORY STRUCTURE WITH DESCRIPTIONS =====
DIRECTORIES = {
    # Raw data storage
    "raw_data": {
        "": "Storage for raw GDELT GKG data files downloaded from GDELT servers",
        "batch_outputs": "Individual batch files (3-month chunks) of raw GDELT data",
        "cache": "Cached downloads to avoid reprocessing the same files"
    },
    
    # Processed data
    "processed_data": {
        "": "Cleaned and processed GDELT data ready for analysis"
    },
    
    # Aggregated data
    "aggregated_data": {
        "": "Data aggregated at different time intervals (15min, hourly, daily)"
    },
    
    # Analysis outputs
    "analysis_results": {
        "": "Results from theme analysis and other analytical processes"
    },
    
    # Visualizations
    "figures": {
        "": "Generated charts, graphs, and other visualizations"
    },
    
    # New directories for model training
    "split_data": {
        "": "Train/test data splits for model training"
    },
    "models": {
        "": "Trained models, scalers, and model checkpoints"
    },
    "model_evaluation": {
        "": "Model evaluation results, metrics, and performance visualizations"
    }
}

# ===== PRE-COMPUTED PATHS FOR CONVENIENCE =====
# These are derived from the DIRECTORIES structure for easy access
PATHS = {
    "BASE_DIR": OUTPUT_DIR,
    "RAW_DATA_DIR": os.path.join(OUTPUT_DIR, "raw_data"),
    "BATCH_DIR": os.path.join(OUTPUT_DIR, "raw_data", "batch_outputs"),
    "CACHE_DIR": os.path.join(OUTPUT_DIR, "raw_data", "cache"),
    "PROCESSED_DIR": os.path.join(OUTPUT_DIR, "processed_data"),
    "AGGREGATED_DIR": os.path.join(OUTPUT_DIR, "aggregated_data"),
    "ANALYSIS_DIR": os.path.join(OUTPUT_DIR, "analysis_results"),
    "FIGURES_DIR": os.path.join(OUTPUT_DIR, "figures"),
    # New paths for model training
    "SPLIT_DATA_DIR": os.path.join(OUTPUT_DIR, "split_data"),
    "MODELS_DIR": os.path.join(OUTPUT_DIR, "models"),
    "MODEL_EVAL_DIR": os.path.join(OUTPUT_DIR, "model_evaluation")
}

# Specific file paths
PARSED_DATA_PATH = os.path.join(PATHS["PROCESSED_DIR"], "processed_gkg_parsed_data.csv")
AGGREGATED_15MIN_PATH = os.path.join(PATHS["AGGREGATED_DIR"], "aggregated_gkg_15min.csv")

# ===== GDELT SETTINGS =====
# Date range for data collection
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2024, 12, 12)  # End date is inclusive

# Location filter
LOCATION_FILTER = "Delhi"

# Request settings
TIMEOUT = 30
MAX_WORKERS = 10

# ===== THEME CATEGORIES =====
# Theme Categories - Used in multiple scripts
THEME_CATEGORIES = {
    'Political': ['ELECTION', 'GOVERN', 'POLIT', 'DEMO', 'LEG', 'VOTE', 'PARLIAMENT', 'PRESIDENT', 'MINISTER'],
    'Economic': ['ECON', 'MARKET', 'TRADE', 'BUSINESS', 'FINANC', 'TAX', 'INVEST', 'GDP', 'INFLATION'],
    'Religious': ['RELIG', 'MUSLIM', 'HINDU', 'SIKH', 'TEMPLE', 'CHURCH', 'MOSQUE', 'FAITH', 'GOD'],
    'Energy': ['ENERGY', 'POWER', 'ELECTRIC', 'OIL', 'GAS', 'COAL', 'FUEL', 'RENEWABLE', 'GRID'],
    'Infrastructure': ['INFRA', 'TRANSPORT', 'CONSTRUCT', 'BUILDING', 'ROAD', 'HIGHWAY', 'RAIL', 'AIRPORT'],
    'Health': ['HEALTH', 'COVID', 'DISEASE', 'PANDEMIC', 'HOSPITAL', 'MEDIC', 'VACCINE', 'DRUG'],
    'Social': ['SOCIAL', 'PROTEST', 'RALLY', 'CELEBR', 'FESTIVAL', 'COMMUNITY', 'SOCIETY', 'PUBLIC', 'CITIZEN'],
    'Environment': ['ENV', 'CLIMATE', 'WEATHER', 'POLLUT', 'WATER', 'GREEN', 'ECOLOGY', 'TEMPERATURE'],
    'Education': ['EDU', 'SCHOOL', 'UNIVERSITY', 'STUDENT', 'LEARNING', 'COLLEGE', 'TEACHER', 'PROFESSOR']
}

# ===== DIRECTORY MANAGEMENT FUNCTIONS =====
def ensure_directory_structure():
    """Create the entire directory structure if it doesn't exist"""
    print(f"Setting up directory structure in {OUTPUT_DIR}")
    
    # Check if OUTPUT_DIR is writable
    if not os.access(os.path.dirname(OUTPUT_DIR), os.W_OK):
        print(f"Error: Cannot write to {os.path.dirname(OUTPUT_DIR)}. Check permissions.")
        return False
    
    # Create main output directory
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"Created main output directory: {OUTPUT_DIR}")
        except Exception as e:
            print(f"Error creating main directory {OUTPUT_DIR}: {e}")
            return False
    
    # Create all subdirectories
    for main_dir, subdirs in DIRECTORIES.items():
        # Create the main category directory
        main_path = os.path.join(OUTPUT_DIR, main_dir)
        if not os.path.exists(main_path):
            try:
                os.makedirs(main_path)
                print(f"Created directory: {main_path}")
            except Exception as e:
                print(f"Error creating directory {main_path}: {e}")
                continue
        
        # Create subdirectories
        for subdir, description in subdirs.items():
            if subdir:  # Skip the empty string key which is just for the main directory description
                sub_path = os.path.join(main_path, subdir)
                if not os.path.exists(sub_path):
                    try:
                        os.makedirs(sub_path)
                        print(f"Created subdirectory: {sub_path}")
                    except Exception as e:
                        print(f"Error creating subdirectory {sub_path}: {e}")
    
    print("Directory structure setup complete")
    return True

def verify_directory_structure():
    """Verify that all required directories exist"""
    missing_dirs = []
    
    # First check the base directory
    if not os.path.exists(OUTPUT_DIR):
        print(f"ERROR: Base output directory doesn't exist: {OUTPUT_DIR}")
        return False
    
    # Check all subdirectories from our structure
    for main_dir, subdirs in DIRECTORIES.items():
        main_path = os.path.join(OUTPUT_DIR, main_dir)
        if not os.path.exists(main_path):
            missing_dirs.append(main_path)
            
        for subdir in subdirs.keys():
            if subdir:  # Skip the empty string key
                sub_path = os.path.join(main_path, subdir)
                if not os.path.exists(sub_path):
                    missing_dirs.append(sub_path)
    
    if missing_dirs:
        print("WARNING: The following directories are missing:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    else:
        print("✓ All required directories exist")
        return True

def test_directory_writing():
    """Test writing to each directory in the structure"""
    print("Testing write access to all directories...")
    
    failed_dirs = []
    
    # Test main output directory
    test_file = os.path.join(OUTPUT_DIR, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write(f"Test write at {datetime.now()}")
        os.remove(test_file)
        print(f"✓ Successfully wrote to {OUTPUT_DIR}")
    except Exception as e:
        print(f"✗ ERROR: Could not write to {OUTPUT_DIR}: {e}")
        failed_dirs.append(OUTPUT_DIR)
    
    # Test all subdirectories
    for main_dir, subdirs in DIRECTORIES.items():
        # Test main category directory
        main_path = os.path.join(OUTPUT_DIR, main_dir)
        test_file = os.path.join(main_path, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write(f"Test write at {datetime.now()}")
            os.remove(test_file)
            print(f"✓ Successfully wrote to {main_path}")
        except Exception as e:
            print(f"✗ ERROR: Could not write to {main_path}: {e}")
            failed_dirs.append(main_path)
        
        # Test subdirectories
        for subdir in subdirs.keys():
            if subdir:  # Skip the empty string key
                sub_path = os.path.join(main_path, subdir)
                test_file = os.path.join(sub_path, "test_write.txt")
                try:
                    with open(test_file, 'w') as f:
                        f.write(f"Test write at {datetime.now()}")
                    os.remove(test_file)
                    print(f"✓ Successfully wrote to {sub_path}")
                except Exception as e:
                    print(f"✗ ERROR: Could not write to {sub_path}: {e}")
                    failed_dirs.append(sub_path)
    
    if failed_dirs:
        print("\n⚠️ WARNING: The following directories have write access issues:")
        for dir_path in failed_dirs:
            print(f"  - {dir_path}")
        return False
    else:
        print("\n✅ All directories are writable.")
        return True

def test_dataframe_write_access():
    """Test writing pandas DataFrames to key directories"""
    import pandas as pd
    
    print("Testing pandas DataFrame write access...")
    test_df = pd.DataFrame({'test': [1, 2, 3]})
    
    failed_paths = []
    for name, path in [
        ("Batch output", PATHS["BATCH_DIR"]), 
        ("Processed output", PATHS["PROCESSED_DIR"]),
        ("Aggregated output", PATHS["AGGREGATED_DIR"])
    ]:
        test_file = os.path.join(path, "test_dataframe.csv")
        try:
            test_df.to_csv(test_file, index=False)
            os.remove(test_file)
            print(f"✓ Successfully wrote DataFrame to {name} directory")
        except Exception as e:
            print(f"✗ ERROR: Failed to write DataFrame to {name} directory: {e}")
            failed_paths.append(path)
    
    if failed_paths:
        print("⚠️ WARNING: DataFrame write test failed for some directories")
        return False
    else:
        print("✅ Successfully wrote DataFrames to all data directories")
        return True

def setup_and_verify():
    """Complete setup and verification process"""
    # First ensure the directory structure exists
    if not ensure_directory_structure():
        print("ERROR: Failed to create directory structure")
        return False
    
    # Verify all directories exist
    if not verify_directory_structure():
        print("ERROR: Directory verification failed")
        return False
    
    # Test writing to all directories
    if not test_directory_writing():
        print("ERROR: Directory write test failed")
        return False
    
    # Test DataFrame writing
    try:
        if not test_dataframe_write_access():
            print("ERROR: DataFrame write test failed")
            return False
    except ImportError:
        print("WARNING: pandas not installed, skipping DataFrame write test")
    
    print("\n✅ Directory setup and verification complete - all systems ready!")
    return True

def get_paths():
    """Return a dictionary of commonly used paths"""
    return PATHS

def print_directory_overview():
    """Print an overview of the directory structure"""
    print("\nDirectory structure overview:")
    for main_dir, subdirs in DIRECTORIES.items():
        print(f"\n{main_dir}/")
        print(f"  {subdirs['']}")  # Print the main directory description
        for subdir, description in subdirs.items():
            if subdir:  # Skip the empty string key
                print(f"  {subdir}/")
                print(f"    {description}")

# For backwards compatibility with existing scripts
ensure_all_directories = ensure_directory_structure
test_directory_access = test_directory_writing

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("GDELT GKG Processing Directory Setup Utility")
    print("--------------------------------------------")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--verify":
            # Only verify, don't create
            verify_directory_structure()
            test_directory_writing()
        elif sys.argv[1] == "--paths":
            # Just print all the paths
            for name, path in get_paths().items():
                print(f"{name}: {path}")
    else:
        # Full setup and verification
        setup_and_verify()
    
    # Always show the directory overview
    print_directory_overview()
