# AI Energy Load Forecasting

## Project Overview
AI Energy Load Forecasting is a comprehensive tool that combines Global Database of Events, Language, and Tone (GDELT) news data with weather information to predict energy load demand. The project implements advanced feature engineering and machine learning techniques to identify patterns and correlations between news events, weather conditions, and energy consumption.

This system leverages natural language processing and time series analysis to extract meaningful insights from news data, particularly focusing on energy-relevant themes such as political, economic, social, and environmental events that might impact energy demand.

## Key Features
- GDELT GKG (Global Knowledge Graph) data gathering and processing
- Automated theme detection and analysis from news data
- Advanced feature engineering for temporal patterns
- Integration of weather and news data for energy load prediction
- Multiple forecasting models (XGBoost, LSTM options)
- Comprehensive visualization and correlation analysis
- Configurable pipeline with centralized settings

## Prerequisites

### Software Requirements
- Python 3.7+
- Required Python Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - xgboost
  - tqdm
  - joblib
  - scipy
  - nltk

### System Requirements
- Minimum 8GB RAM recommended (16GB+ for large datasets)
- At least 10GB free disk space for data storage
- Internet connection for GDELT data gathering

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-Energy-Load-New.git
   cd AI-Energy-Load-New
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure project paths in `config.py` if necessary.

## Project Structure
- `GDELT_gkg/`: Scripts for gathering and processing GDELT data
- `Weather_Energy/`: Weather and energy load data processing
- `feature_engineering/`: Advanced feature extraction modules
- `Correlation_and_preprocessing/`: Data merge and correlation analysis
- `Model_pedictor/`: Machine learning models for prediction
- `config.py`: Centralized configuration settings

## Running the Project

Follow these steps in the correct order to execute the project pipeline:

### 1. Setup and Configuration
First, verify the project configuration and directory structure:
```bash
python config.py
```

This creates necessary directories and verifies write access.

### 2. GDELT Data Collection and Processing
```bash
python GDELT_gkg/gkg_data_gathering.py
python GDELT_gkg/gkg_data_sparsing.py
python GDELT_gkg/gkg_data_aggregation.py
python GDELT_gkg/gkg_data_theme_analysis.py
```

### 3. Feature Engineering and Data Integration
```bash
python feature_engineering/theme_evolution.py
python feature_engineering/theme_occurence.py
python feature_engineering/temporal_effects.py
python Correlation_and_preprocessing/merge.py
```

### 4. Correlation Analysis
```bash
python Correlation_and_preprocessing/Correalation.py
```

### 5. Model Training and Prediction
```bash
python Model_pedictor/preparation.py
python Model_pedictor/xg_boost_model.py
```

## Output
The project generates various outputs in the `OUTPUT_DIR` directory (configurable in `config.py`):
- Processed and aggregated GDELT data
- Feature engineering results
- Correlation analysis visualizations
- Model predictions and evaluation metrics
- Trained models and scalers

## Customization
- Modify date ranges and location filters in `config.py`
- Adjust feature engineering parameters in the `FEATURE_ENGINEERING` section of `config.py`
- Configure model hyperparameters in the respective model files

## Troubleshooting
- Ensure all directories have proper write permissions
- For memory issues, try reducing chunk sizes in data processing steps
- Check log files in the `logs` directory for detailed error information

## License
[Your License Information]

## Contributors
[Your Contributor Information]
