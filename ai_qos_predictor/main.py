#!/usr/bin/env python3
"""
AI QoS Predictor for Mobile Video Streaming

This is the main entry point for the AI QoS predictor project. It provides a unified
interface for data processing, model training, evaluation, and real-time prediction.

Usage:
    python main.py process-data --dataset 5g_kpi
    python main.py train --model lstm
    python main.py evaluate --models lstm transformer
    python main.py predict --model lstm --interface eth0
    python main.py visualize --model-path models/saved/model.h5
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_DIR = Path(__file__).parent
PREPROCESSING_DIR = PROJECT_DIR / "preprocessing"
MODELS_DIR = PROJECT_DIR / "models"
EVALUATION_DIR = PROJECT_DIR / "evaluation"
VISUALIZATION_DIR = PROJECT_DIR / "visualization"

def process_data(args):
    """
    Process datasets for model training
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Processing {args.dataset} dataset")
    
    if args.dataset == '5g_kpi':
        script_path = PREPROCESSING_DIR / "process_5g_dataset.py"
    elif args.dataset == 'youtube':
        script_path = PREPROCESSING_DIR / "process_youtube_dataset.py"
    elif args.dataset == 'all':
        # Process all datasets
        for script in PREPROCESSING_DIR.glob("process_*.py"):
            logger.info(f"Running {script.name}")
            subprocess.run([sys.executable, str(script)], check=True)
        
        # Run feature extraction
        logger.info("Running feature extraction")
        feature_script = PREPROCESSING_DIR / "feature_extraction.py"
        subprocess.run([sys.executable, str(feature_script)], check=True)
        
        logger.info("All datasets processed successfully")
        return
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return
    
    # Run the processing script
    if script_path.exists():
        subprocess.run([sys.executable, str(script_path)], check=True)
        
        # Run feature extraction if requested
        if args.extract_features:
            logger.info("Running feature extraction")
            feature_script = PREPROCESSING_DIR / "feature_extraction.py"
            subprocess.run([sys.executable, str(feature_script)], check=True)
        
        logger.info(f"{args.dataset} dataset processed successfully")
    else:
        logger.error(f"Processing script not found for {args.dataset}")

def train_model(args):
    """
    Train a model for QoS prediction
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Training {args.model} model")
    
    # Use the unified train_model.py script
    script_path = MODELS_DIR / "train_model.py"
    
    # Run the training script
    if script_path.exists():
        cmd = [sys.executable, str(script_path)]
        
        # Add model type if provided
        if args.model:
            cmd.extend(['--model-type', args.model])
        
        # Add dataset if provided
        if args.dataset:
            cmd.extend(['--dataset', args.dataset])
        
        # Add hyperparameters if provided
        if args.config:
            cmd.extend(['--config', args.config])
        
        subprocess.run(cmd, check=True)
        logger.info(f"{args.model} model trained successfully")
    else:
        logger.error(f"Training script not found: {script_path}")

def evaluate_models(args):
    """
    Evaluate and compare models
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Evaluating models: {', '.join(args.models)}")
    
    # Run the evaluation script
    script_path = EVALUATION_DIR / "model_comparison.py"
    
    if script_path.exists():
        cmd = [sys.executable, str(script_path)]
        
        # Add model paths if provided
        if args.model_paths:
            cmd.extend(['--model-paths', args.model_paths])
        
        # Add models to compare
        cmd.extend(['--models'] + args.models)
        
        subprocess.run(cmd, check=True)
        logger.info("Model evaluation completed successfully")
    else:
        logger.error("Evaluation script not found")

def run_prediction(args):
    """
    Run real-time prediction
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Running real-time prediction with {args.model} model")
    
    # Run the prediction script
    script_path = PROJECT_DIR / "real_time_predictor.py"
    
    if script_path.exists():
        cmd = [sys.executable, str(script_path)]
        
        # Add model path if provided
        if args.model_path:
            cmd.extend(['--model', args.model_path])
        elif args.model:
            # Look for the specified model type in both the current models directory and the parent models directory
            model_files = list(MODELS_DIR.glob(f"*{args.model}*.pt"))
            parent_models_dir = Path("../models")
            parent_model_files = list(parent_models_dir.glob(f"*{args.model}*.pt"))
            
            # Combine the lists
            all_model_files = model_files + parent_model_files
            
            if all_model_files:
                # Use the most recent model of the specified type
                model_path = str(sorted(all_model_files, key=os.path.getmtime)[-1])
                cmd.extend(['--model', model_path])
            else:
                logger.error(f"No models found. Please train a model first or specify a model path.")
                return
        
        # Add interface if provided
        if args.interface:
            cmd.extend(['--interface', args.interface])
        
        # Add buffer size if provided
        if args.buffer_size:
            cmd.extend(['--buffer-size', str(args.buffer_size)])
        
        # Add update interval if provided
        if args.update_interval:
            cmd.extend(['--update-interval', str(args.update_interval)])
        
        subprocess.run(cmd, check=True)
    else:
        logger.error("Prediction script not found")

def visualize_predictions(args):
    """
    Generate visualizations comparing predicted vs real values
    
    Args:
        args: Command-line arguments
    """
    logger.info("Generating visualizations for model predictions")
    
    # Run the visualization script
    script_path = VISUALIZATION_DIR / "compare_predictions.py"
    
    if script_path.exists():
        cmd = [sys.executable, str(script_path)]
        
        # Add model path if provided
        if args.model_path:
            cmd.extend(['--model-path', args.model_path])
        elif args.model:
            # Look for the specified model type in both the current models directory and the parent models directory
            model_files = list(MODELS_DIR.glob(f"*{args.model}*.pt"))
            parent_models_dir = Path("../models")
            parent_model_files = list(parent_models_dir.glob(f"*{args.model}*.pt"))
            
            # Combine the lists
            all_model_files = model_files + parent_model_files
            
            if all_model_files:
                # Use the most recent model of the specified type
                model_path = str(sorted(all_model_files, key=os.path.getmtime)[-1])
                cmd.extend(['--model-path', model_path])
            else:
                logger.error(f"No models found. Please train a model first or specify a model path.")
                return
        
        # Add dataset if provided
        if args.dataset:
            cmd.extend(['--dataset', args.dataset])
        
        # Add output directory if provided
        if args.output_dir:
            cmd.extend(['--output-dir', args.output_dir])
        
        subprocess.run(cmd, check=True)
        logger.info("Visualization generation completed successfully")
    else:
        logger.error(f"Visualization script not found: {script_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI QoS Predictor for Mobile Video Streaming')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process data command
    process_parser = subparsers.add_parser('process-data', help='Process datasets for model training')
    process_parser.add_argument('--dataset', choices=['5g_kpi', 'youtube', 'all'], default='all',
                               help='Dataset to process')
    process_parser.add_argument('--extract-features', action='store_true',
                               help='Run feature extraction after processing')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a model for QoS prediction')
    train_parser.add_argument('--model', choices=['lstm', 'transformer'], default='lstm',
                             help='Type of model to train')
    train_parser.add_argument('--dataset', choices=['5g_kpi', 'youtube', 'all'], default='5g_kpi',
                             help='Dataset to use for training')
    train_parser.add_argument('--config', type=str,
                             help='Path to model configuration JSON file')
    
    # Evaluate models command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate and compare models')
    evaluate_parser.add_argument('--models', nargs='+', choices=['lstm', 'transformer'], required=True,
                                help='Models to evaluate')
    evaluate_parser.add_argument('--model-paths', type=str,
                                help='Path to JSON file with model paths')
    
    # Run prediction command
    predict_parser = subparsers.add_parser('predict', help='Run real-time prediction')
    predict_parser.add_argument('--model', choices=['lstm', 'transformer'], default='lstm',
                               help='Type of model to use')
    predict_parser.add_argument('--model-path', type=str,
                               help='Path to the trained model')
    predict_parser.add_argument('--interface', type=str, default='eth0',
                               help='Network interface to monitor')
    predict_parser.add_argument('--buffer-size', type=int, default=30,
                               help='Number of seconds to keep in the buffer')
    predict_parser.add_argument('--update-interval', type=float, default=1.0,
                               help='Seconds between updates')
    
    # Visualize predictions command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualizations comparing predicted vs real values')
    visualize_parser.add_argument('--model', choices=['lstm', 'transformer'],
                                 help='Type of model to use')
    visualize_parser.add_argument('--model-path', type=str,
                                 help='Path to the trained model')
    visualize_parser.add_argument('--dataset', default='5g_kpi',
                                 help='Dataset to use for visualization')
    visualize_parser.add_argument('--output-dir', type=str,
                                 help='Directory to save visualizations')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == 'process-data':
        process_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_models(args)
    elif args.command == 'predict':
        run_prediction(args)
    elif args.command == 'visualize':
        visualize_predictions(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()