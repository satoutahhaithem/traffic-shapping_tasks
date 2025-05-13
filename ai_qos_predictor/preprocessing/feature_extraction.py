#!/usr/bin/env python3
"""
Feature Extraction for QoS Prediction

This script combines processed data from different datasets,
extracts and normalizes features, creates windowed sequences
for time-series prediction, and prepares train/validation/test splits.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

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
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURES_DIR = PROJECT_DIR / "data" / "features"
SEQUENCE_LENGTH = 10  # Number of time steps to use for prediction (10 seconds)
PREDICTION_HORIZON = 3  # Number of seconds ahead to predict

# Core network features that should be present in all datasets
CORE_FEATURES = [
    'throughput_down',  # Downlink throughput in Mbps
    'throughput_up',    # Uplink throughput in Mbps
    'rtt',              # Round-trip time in ms
    'jitter',           # Jitter in ms
    'packet_loss'       # Packet loss percentage
]

# Signal quality features (may not be present in all datasets)
SIGNAL_FEATURES = [
    'sinr',             # Signal-to-Interference-plus-Noise Ratio
    'rsrp',             # Reference Signal Received Power
    'rsrq'              # Reference Signal Received Quality
]

# Network QoS features to predict (future values of the same features)
NETWORK_QOS_TARGETS = [
    'throughput_down',  # Future downlink throughput
    'rtt',              # Future round-trip time
    'packet_loss'       # Future packet loss
]

def load_dataset(dataset_name):
    """Load a processed dataset"""
    parquet_file = PROCESSED_DIR / dataset_name / f"{dataset_name}_processed.parquet"
    csv_file = PROCESSED_DIR / dataset_name / f"{dataset_name}_processed.csv"
    
    logger.info(f"Looking for dataset in {PROCESSED_DIR / dataset_name}")
    
    if parquet_file.exists():
        logger.info(f"Loading {dataset_name} dataset from {parquet_file}")
        return pd.read_parquet(parquet_file)
    elif csv_file.exists():
        logger.info(f"Loading {dataset_name} dataset from {csv_file}")
        return pd.read_csv(csv_file)
    else:
        logger.warning(f"Dataset {dataset_name} not found at {csv_file} or {parquet_file}")
        return None

def align_features(dataframes, feature_sets):
    """
    Align features across different datasets
    
    Args:
        dataframes: Dictionary of {dataset_name: dataframe}
        feature_sets: List of feature sets to align
        
    Returns:
        Dictionary of aligned dataframes
    """
    aligned_dfs = {}
    
    # Identify common features across all datasets
    all_features = []
    for feature_set in feature_sets:
        all_features.extend(feature_set)
    
    for name, df in dataframes.items():
        if df is None:
            continue
            
        # Check which features are available in this dataset
        available_features = [f for f in all_features if f in df.columns]
        missing_features = [f for f in all_features if f not in df.columns]
        
        if missing_features:
            logger.info(f"Dataset {name} is missing features: {missing_features}")
            
            # Add missing features with NaN values
            for feature in missing_features:
                df[feature] = np.nan
        
        # Ensure timestamp is in datetime format
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        aligned_dfs[name] = df
    
    return aligned_dfs

def normalize_features(dataframes):
    """
    Normalize numeric features and encode categorical features
    
    Args:
        dataframes: Dictionary of {dataset_name: dataframe}
        
    Returns:
        Dictionary of normalized dataframes and the fitted transformers
    """
    normalized_dfs = {}
    transformers = {}
    
    # Combine all dataframes to fit the transformers
    combined_df = pd.concat(dataframes.values(), ignore_index=True)
    
    # Identify numeric and categorical features
    numeric_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove timestamp from features to normalize
    if 'timestamp' in numeric_features:
        numeric_features.remove('timestamp')
    
    # Fit standard scaler on numeric features
    scaler = StandardScaler()
    if numeric_features:
        scaler.fit(combined_df[numeric_features].fillna(0))
        transformers['scaler'] = scaler
    
    # Fit one-hot encoder on categorical features
    # Use sparse_output=False instead of sparse=False for newer scikit-learn versions
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if categorical_features:
        encoder.fit(combined_df[categorical_features].fillna('unknown'))
        transformers['encoder'] = encoder
    
    # Apply transformations to each dataset
    for name, df in dataframes.items():
        df_copy = df.copy()
        
        # Apply standard scaling to numeric features
        if numeric_features:
            df_copy[numeric_features] = scaler.transform(df_copy[numeric_features].fillna(0))
        
        # Apply one-hot encoding to categorical features
        if categorical_features:
            encoded_features = encoder.transform(df_copy[categorical_features].fillna('unknown'))
            encoded_df = pd.DataFrame(
                encoded_features, 
                columns=encoder.get_feature_names_out(categorical_features),
                index=df_copy.index
            )
            
            # Drop original categorical columns and add encoded ones
            df_copy = df_copy.drop(columns=categorical_features)
            df_copy = pd.concat([df_copy, encoded_df], axis=1)
        
        normalized_dfs[name] = df_copy
    
    return normalized_dfs, transformers

def create_sequences(df, sequence_length, prediction_horizon, features, targets):
    """
    Create windowed sequences for time-series prediction
    
    Args:
        df: DataFrame with time-series data
        sequence_length: Number of time steps in each input sequence
        prediction_horizon: Number of time steps ahead to predict
        features: List of feature columns to use as input
        targets: List of target columns to predict
        
    Returns:
        X: Input sequences of shape (n_samples, sequence_length, n_features)
        y: Target values of shape (n_samples, n_targets)
    """
    X, y = [], []
    
    # Ensure all required columns exist
    missing_features = [f for f in features if f not in df.columns]
    missing_targets = [t for t in targets if t not in df.columns]
    
    if missing_features or missing_targets:
        logger.warning(f"Missing features: {missing_features}, missing targets: {missing_targets}")
        return np.array([]), np.array([])
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Group by scenario/experiment_id if available
    groupby_cols = []
    for col in ['scenario', 'experiment_id', 'mobility', 'service']:
        if col in df.columns:
            groupby_cols.append(col)
    
    if groupby_cols:
        groups = df.groupby(groupby_cols)
    else:
        # If no grouping columns, treat the entire dataframe as one group
        groups = [(None, df)]
    
    for _, group_df in groups:
        # Skip groups that are too small
        if len(group_df) < sequence_length + prediction_horizon:
            continue
            
        # Get feature and target values
        feature_values = group_df[features].values
        target_values = group_df[targets].values
        
        for i in range(len(group_df) - sequence_length - prediction_horizon + 1):
            # Input sequence: current window of features
            X.append(feature_values[i:i+sequence_length])
            
            # Target value: future values of target features
            y.append(target_values[i+sequence_length+prediction_horizon-1])
    
    return np.array(X), np.array(y)

def prepare_data_for_training(normalized_dfs, input_features, target_features):
    """
    Prepare data for model training by creating sequences and train/val/test splits
    
    Args:
        normalized_dfs: Dictionary of normalized dataframes
        input_features: List of feature columns to use as input
        target_features: List of target columns to predict
        
    Returns:
        Dictionary with train/val/test data and metadata
    """
    # Prepare data for each dataset
    dataset_sequences = {}
    
    for name, df in normalized_dfs.items():
        # Get available features and targets for this dataset
        available_features = [f for f in input_features if f in df.columns]
        available_targets = [t for t in target_features if t in df.columns]
        
        if not available_features or not available_targets:
            logger.warning(f"Dataset {name} has insufficient features or targets, skipping")
            continue
        
        # Create sequences
        X, y = create_sequences(
            df, 
            SEQUENCE_LENGTH, 
            PREDICTION_HORIZON, 
            available_features, 
            available_targets
        )
        
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"No sequences created for dataset {name}, skipping")
            continue
        
        # Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        dataset_sequences[name] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'features': available_features,
            'targets': available_targets
        }
        
        logger.info(f"Dataset {name} prepared: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test sequences")
    
    return dataset_sequences

def main():
    """Main function to extract features and prepare data for model training"""
    logger.info("Starting feature extraction and preparation")
    
    # Create features directory
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load processed datasets
    datasets = {
        '5g_kpi': load_dataset('5g_kpi'),
        'youtube_mobile': load_dataset('youtube_mobile'),
        # Add other datasets as they become available
    }
    
    # Filter out None values
    datasets = {k: v for k, v in datasets.items() if v is not None}
    
    if not datasets:
        logger.error("No datasets available for feature extraction")
        return
    
    # Align features across datasets
    logger.info("Aligning features across datasets")
    aligned_dfs = align_features(datasets, [CORE_FEATURES, SIGNAL_FEATURES])
    
    # Normalize features
    logger.info("Normalizing features")
    normalized_dfs, transformers = normalize_features(aligned_dfs)
    
    # Save transformers for later use
    with open(FEATURES_DIR / 'transformers.pkl', 'wb') as f:
        pickle.dump(transformers, f)
    
    # Prepare data for model training
    logger.info("Preparing data for model training")
    
    # Use all available features as input
    all_input_features = CORE_FEATURES + SIGNAL_FEATURES
    
    # Use network QoS features as targets (predict future network conditions)
    dataset_sequences = prepare_data_for_training(
        normalized_dfs,
        all_input_features,
        NETWORK_QOS_TARGETS
    )
    
    # Save prepared data
    for name, data in dataset_sequences.items():
        output_file = FEATURES_DIR / f"{name}_sequences.npz"
        np.savez(
            output_file,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            X_test=data['X_test'],
            y_test=data['y_test']
        )
        
        # Save metadata
        metadata = {
            'features': data['features'],
            'targets': data['targets'],
            'sequence_length': SEQUENCE_LENGTH,
            'prediction_horizon': PREDICTION_HORIZON
        }
        
        with open(FEATURES_DIR / f"{name}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved prepared data for {name} to {output_file}")
    
    logger.info("Feature extraction and preparation completed")

if __name__ == "__main__":
    main()