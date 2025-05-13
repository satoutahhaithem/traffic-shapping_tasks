#!/usr/bin/env python3
"""
Process the YouTube Mobile Streaming Dataset

This script processes the YouTube mobile streaming dataset from Figshare,
which contains packet captures and video QoE metrics from experiments
that replay real 3G/4G drive traces in a controlled lab environment.

Dataset source: https://figshare.com/articles/dataset/YouTube_Dataset_on_Mobile_Streaming_for_Internet_Traffic_Modeling_Network_Management_and_Streaming_Analysis/19096823
"""

import os
import pandas as pd
import numpy as np
import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("../data/raw/youtube_mobile")
PROCESSED_DIR = Path("../data/processed/youtube_mobile")
PCAP_DIR = DATA_DIR / "pcap"
STATS_DIR = DATA_DIR / "stats"

def check_dataset_availability():
    """Check if the dataset is available and provide download instructions if not"""
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.warning(f"YouTube mobile streaming dataset not found in {DATA_DIR}")
        logger.info("""
        Please download the dataset manually from Figshare:
        https://figshare.com/articles/dataset/YouTube_Dataset_on_Mobile_Streaming_for_Internet_Traffic_Modeling_Network_Management_and_Streaming_Analysis/19096823
        
        Due to the large size of this dataset (60+ GB), automatic download is not implemented.
        
        After downloading, extract the files to the following structure:
        - {DATA_DIR}/pcap/    # For packet capture files
        - {DATA_DIR}/stats/   # For YouTube stats-for-nerds data
        - {DATA_DIR}/metadata.json  # Experiment metadata
        """)
        return False
    return True

def extract_pcap_metrics(pcap_file):
    """
    Extract network metrics from a PCAP file using tshark
    
    This function uses tshark (Wireshark CLI) to extract:
    - Throughput per second
    - RTT from TCP handshakes and ACK pairs
    - Packet loss estimation
    """
    logger.info(f"Processing PCAP file: {pcap_file}")
    
    # Create a temporary output file
    output_file = PROCESSED_DIR / "temp" / f"{pcap_file.stem}_metrics.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract per-second throughput
        throughput_cmd = [
            "tshark", "-r", str(pcap_file), 
            "-q", "-z", "io,stat,1", 
            "-T", "fields", 
            "-e", "interval", "-e", "sum(ip.len)"
        ]
        
        # Run the command and capture output
        result = subprocess.run(throughput_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error extracting throughput from {pcap_file}: {result.stderr}")
            return None
        
        # Parse the output into a DataFrame
        lines = result.stdout.strip().split('\n')
        data = []
        
        for line in lines:
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    interval = parts[0]
                    bytes_per_sec = parts[1]
                    data.append({
                        'timestamp': interval,
                        'throughput_down': int(bytes_per_sec) * 8 / 1000000  # Convert bytes to Mbps
                    })
        
        df = pd.DataFrame(data)
        
        # Extract RTT using TCP analysis
        # This is a simplified approach - in a real implementation, you would need
        # more sophisticated analysis to accurately extract RTT from PCAP files
        rtt_cmd = [
            "tshark", "-r", str(pcap_file),
            "-Y", "tcp.analysis.ack_rtt",
            "-T", "fields",
            "-e", "frame.time_epoch",
            "-e", "tcp.analysis.ack_rtt"
        ]
        
        result = subprocess.run(rtt_cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            rtt_data = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        time_epoch = float(parts[0])
                        rtt_value = float(parts[1]) * 1000  # Convert to ms
                        timestamp = datetime.fromtimestamp(time_epoch).strftime('%Y-%m-%d %H:%M:%S')
                        rtt_data.append({
                            'timestamp': timestamp,
                            'rtt': rtt_value
                        })
            
            rtt_df = pd.DataFrame(rtt_data)
            
            # Convert timestamp to datetime
            rtt_df['timestamp'] = pd.to_datetime(rtt_df['timestamp'])
            
            # Resample to 1-second intervals
            rtt_df = rtt_df.set_index('timestamp')
            rtt_df = rtt_df.resample('1S').mean()
            rtt_df = rtt_df.reset_index()
            
            # Merge with throughput data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = pd.merge(df, rtt_df, on='timestamp', how='left')
        
        # Save the extracted metrics
        df.to_csv(output_file, index=False)
        logger.info(f"Saved PCAP metrics to {output_file}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing PCAP file {pcap_file}: {e}")
        return None

def process_stats_file(stats_file):
    """Process a YouTube stats-for-nerds JSON file to extract QoE metrics"""
    logger.info(f"Processing stats file: {stats_file}")
    
    try:
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)
        
        # Extract relevant metrics
        data = []
        
        for entry in stats_data:
            if 'timestamp' in entry and 'stats' in entry:
                timestamp = entry['timestamp']
                stats = entry['stats']
                
                # Extract quality level, buffer health, etc.
                quality_level = stats.get('quality', 'unknown')
                resolution = stats.get('resolution', 'unknown')
                buffer_health = stats.get('bufferHealth', 0)
                
                # Check for stall events
                stall_event = 0
                if buffer_health <= 0.5:  # Consider it a potential stall if buffer is low
                    stall_event = 1
                
                data.append({
                    'timestamp': timestamp,
                    'quality_level': quality_level,
                    'resolution': resolution,
                    'buffer_health': buffer_health,
                    'stall_event': stall_event
                })
        
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resample to 1-second intervals
        df = df.set_index('timestamp')
        
        # For numeric columns, use mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].resample('1S').mean()
        
        # For categorical columns, use most frequent value
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        df_categorical = df[categorical_cols].resample('1S').apply(lambda x: x.mode().iloc[0] if not x.empty else None)
        
        # Combine the resampled dataframes
        df_resampled = pd.concat([df_numeric, df_categorical], axis=1)
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    except Exception as e:
        logger.error(f"Error processing stats file {stats_file}: {e}")
        return None

def align_and_merge_data(pcap_metrics, stats_metrics, experiment_id):
    """Align and merge network metrics with QoE metrics"""
    if pcap_metrics is None or stats_metrics is None:
        logger.warning(f"Missing data for experiment {experiment_id}, skipping merge")
        return None
    
    try:
        # Ensure timestamps are in datetime format
        pcap_metrics['timestamp'] = pd.to_datetime(pcap_metrics['timestamp'])
        stats_metrics['timestamp'] = pd.to_datetime(stats_metrics['timestamp'])
        
        # Merge on timestamp
        merged_df = pd.merge_asof(
            pcap_metrics.sort_values('timestamp'),
            stats_metrics.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('1s')
        )
        
        # Add experiment ID
        merged_df['experiment_id'] = experiment_id
        
        return merged_df
    
    except Exception as e:
        logger.error(f"Error aligning data for experiment {experiment_id}: {e}")
        return None

def process_dataset():
    """Process the entire YouTube mobile streaming dataset"""
    if not check_dataset_availability():
        return
    
    # Create processed directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all PCAP files
    pcap_files = list(PCAP_DIR.glob("*.pcap"))
    
    if not pcap_files:
        logger.warning(f"No PCAP files found in {PCAP_DIR}")
        return
    
    logger.info(f"Found {len(pcap_files)} PCAP files to process")
    
    # Process each experiment
    all_data = []
    
    for pcap_file in pcap_files:
        experiment_id = pcap_file.stem
        
        # Find corresponding stats file
        stats_file = STATS_DIR / f"{experiment_id}.json"
        
        if not stats_file.exists():
            logger.warning(f"Stats file not found for experiment {experiment_id}, skipping")
            continue
        
        # Process PCAP file
        pcap_metrics = extract_pcap_metrics(pcap_file)
        
        # Process stats file
        stats_metrics = process_stats_file(stats_file)
        
        # Align and merge data
        merged_data = align_and_merge_data(pcap_metrics, stats_metrics, experiment_id)
        
        if merged_data is not None:
            all_data.append(merged_data)
    
    if not all_data:
        logger.warning("No data was successfully processed")
        return
    
    # Combine all processed data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save the combined processed data
    output_file = PROCESSED_DIR / "youtube_mobile_processed.csv"
    combined_df.to_csv(output_file, index=False)
    
    # Also save a parquet file for more efficient storage and reading
    parquet_file = PROCESSED_DIR / "youtube_mobile_processed.parquet"
    combined_df.to_parquet(parquet_file, index=False)
    
    logger.info(f"Processed data saved to {output_file} and {parquet_file}")
    logger.info(f"Total records: {len(combined_df)}")
    
    return combined_df

def main():
    """Main function to process the YouTube mobile streaming dataset"""
    logger.info("Starting YouTube mobile streaming dataset processing")
    
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
    
    logger.info("YouTube mobile streaming dataset processing completed")

if __name__ == "__main__":
    main()