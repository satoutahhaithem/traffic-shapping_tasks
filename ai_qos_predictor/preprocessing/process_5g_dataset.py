#!/usr/bin/env python3
"""
Process the 5G KPI Dataset ("Beyond Throughput")

This script downloads and processes the 5G KPI dataset, which contains
network metrics from a phone inside a car driving a 7km urban loop.
The dataset includes throughput, RTT, jitter, packet loss, and signal metrics.

Dataset source: https://github.com/acmmmsys/2020-5Gdataset
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the absolute path to the project directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Constants
DATASET_URL = "https://github.com/acmmmsys/2020-5Gdataset/archive/refs/heads/master.zip"
DATA_DIR = PROJECT_DIR / "data" / "raw" / "5g_kpi"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "5g_kpi"
FEATURES = [
    'timestamp', 'throughput_down', 'throughput_up', 'rtt', 
    'jitter', 'packet_loss', 'sinr', 'rsrp', 'cell_id'
]

def download_dataset():
    """Download the 5G KPI dataset if not already present"""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    dataset_zip = DATA_DIR / "5g_dataset.zip"
    inner_zip = DATA_DIR / "2020-5Gdataset-master" / "5G-production-dataset.zip"
    
    if not dataset_zip.exists() and not (DATA_DIR / "5G-production-dataset").exists():
        logger.info(f"Downloading 5G KPI dataset from {DATASET_URL}")
        try:
            response = requests.get(DATASET_URL)
            response.raise_for_status()
            
            with open(dataset_zip, 'wb') as f:
                f.write(response.content)
                
            # Extract the outer zip file
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            
            # Extract the inner zip file if it exists
            if inner_zip.exists():
                with zipfile.ZipFile(inner_zip, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                    
            logger.info(f"Dataset downloaded and extracted to {DATA_DIR}")
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.info("Please download the dataset manually from GitHub and place it in the data directory")
    else:
        if dataset_zip.exists():
            logger.info(f"Dataset already exists at {dataset_zip}")
        if (DATA_DIR / "5G-production-dataset").exists():
            logger.info(f"Extracted dataset found at {DATA_DIR / '5G-production-dataset'}")

def map_column_names(df):
    """Map original column names to standardized feature names"""
    # This mapping is based on the actual column names in the dataset
    column_mapping = {
        'Timestamp': 'timestamp',
        'DL_bitrate': 'throughput_down',
        'UL_bitrate': 'throughput_up',
        'PINGAVG': 'rtt',
        'PINGSTDEV': 'jitter',
        'PINGLOSS': 'packet_loss',
        'SNR': 'sinr',
        'RSRP': 'rsrp',
        'CellID': 'cell_id',
        'RSRQ': 'rsrq'  # Added RSRQ column
    }
    
    # Create a mapping for columns that exist in the dataframe
    valid_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    
    # Rename columns
    df = df.rename(columns=valid_mapping)
    
    return df

def process_file(file_path):
    """Process a single CSV file from the dataset"""
    logger.info(f"Processing {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Map column names to standardized feature names
        df = map_column_names(df)
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns:
            # The timestamp format is YYYY.MM.DD_HH.MM.SS
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y.%m.%d_%H.%M.%S')
        
        # Convert all columns that should be numeric
        numeric_cols = ['throughput_down', 'throughput_up', 'rtt', 'jitter', 
                        'packet_loss', 'sinr', 'rsrp', 'rsrq']
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert to string first to handle any non-string values
                df[col] = df[col].astype(str)
                # Replace '-' with NaN
                df[col] = df[col].replace('-', np.nan)
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with forward fill, then backward fill
        df = df.ffill().bfill()
        
        # Skip resampling as it's causing issues with mixed data types
        # Instead, just ensure the data is sorted by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None

def process_dataset():
    """Process all CSV files in the dataset"""
    if not PROCESSED_DIR.exists():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files in the dataset directory (recursive search)
    csv_files = []
    
    # Look in the 5G-production-dataset directory
    production_dir = DATA_DIR / "5G-production-dataset"
    if production_dir.exists():
        logger.info(f"Looking for CSV files in {production_dir}")
        csv_files.extend(list(production_dir.glob("**/*.csv")))
    
    # Also look directly in the data directory
    csv_files.extend(list(DATA_DIR.glob("*.csv")))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {DATA_DIR}")
        # List directories to debug
        logger.info(f"Contents of {DATA_DIR}:")
        for item in DATA_DIR.iterdir():
            logger.info(f"  {item}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file and store the results
    all_data = []
    for file_path in csv_files:
        df = process_file(file_path)
        if df is not None:
            # Add a column for the file name (to identify different drives/scenarios)
            df['scenario'] = file_path.stem
            
            # Add columns for driving/static and service type based on the file path
            path_str = str(file_path)
            if "Driving" in path_str:
                df['mobility'] = "Driving"
            elif "Static" in path_str:
                df['mobility'] = "Static"
            else:
                df['mobility'] = "Unknown"
                
            if "Amazon_Prime" in path_str:
                df['service'] = "Amazon_Prime"
            elif "Netflix" in path_str:
                df['service'] = "Netflix"
            elif "Download" in path_str:
                df['service'] = "Download"
            else:
                df['service'] = "Unknown"
            
            all_data.append(df)
    
    if not all_data:
        logger.warning("No data was successfully processed")
        return
    
    # Combine all processed data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create the processed directory if it doesn't exist
    if not PROCESSED_DIR.exists():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the combined processed data to CSV
    output_file = PROCESSED_DIR / "5g_kpi_processed.csv"
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Processed data saved to {output_file}")
    
    # Try to save as parquet, but catch any errors
    try:
        parquet_file = PROCESSED_DIR / "5g_kpi_processed.parquet"
        combined_df.to_parquet(parquet_file, index=False)
        logger.info(f"Processed data also saved to {parquet_file}")
    except Exception as e:
        logger.warning(f"Could not save to parquet format: {e}")
        logger.info("Continuing with CSV format only")
    
    logger.info(f"Total records: {len(combined_df)}")
    
    return combined_df

def main():
    """Main function to download and process the dataset"""
    logger.info("Starting 5G KPI dataset processing")
    
    # Download the dataset
    download_dataset()
    
    # Process the dataset
    df = process_dataset()
    
    if df is not None:
        # Display some statistics
        logger.info("\nDataset statistics:")
        logger.info(f"Shape: {df.shape}")
        logger.info("\nFeature summary:")
        
        # Print summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats = df[col].describe()
            logger.info(f"{col}:")
            logger.info(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}")
    
    logger.info("5G KPI dataset processing completed")

if __name__ == "__main__":
    main()