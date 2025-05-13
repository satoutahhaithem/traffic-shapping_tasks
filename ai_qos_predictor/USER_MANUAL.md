# AI QoS Predictor: User Manual

This comprehensive user manual provides detailed instructions on how to use the AI QoS Predictor system for mobile video streaming.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Command-Line Interface](#command-line-interface)
4. [Data Processing](#data-processing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Real-Time Prediction](#real-time-prediction)
8. [Integration with Video Players](#integration-with-video-players)
9. [Advanced Configuration](#advanced-configuration)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

## System Overview

The AI QoS Predictor is a machine learning system designed to predict future network conditions for mobile video streaming. By analyzing patterns in network metrics like throughput, round-trip time (RTT), and packet loss, the system can anticipate network degradation before it affects video quality.

### Key Components

1. **Data Processing Pipeline**: Processes raw network measurement datasets into a format suitable for machine learning.
2. **Feature Extraction**: Creates time-series sequences for model training.
3. **LSTM Model**: Deep learning model for predicting future network conditions.
4. **Evaluation Tools**: Metrics and visualizations for assessing model performance.
5. **Real-Time Predictor**: System for making predictions on live network data.
6. **Video Player Integration**: Example code for integrating predictions with video players.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step-by-Step Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ai_qos_predictor.git
   cd ai_qos_predictor
   ```

2. **Run the Setup Script**

   The setup script automates the installation process:

   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This script will:
   - Check Python version
   - Create a virtual environment
   - Install dependencies
   - Create the directory structure
   - Make scripts executable

3. **Manual Installation (Alternative)**

   If you prefer to install manually:

   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Create directory structure
   mkdir -p data/raw data/processed data/features
   mkdir -p models/saved models/results
   mkdir -p evaluation

   # Make scripts executable
   chmod +x main.py
   chmod +x models/train_model.py
   ```

## Command-Line Interface

The system provides a unified command-line interface through the `main.py` script:

```bash
./main.py <command> [options]
```

Available commands:

- `process-data`: Process datasets for model training
- `train`: Train a model for QoS prediction
- `evaluate`: Evaluate and compare models
- `predict`: Run real-time prediction

### Getting Help

To see available commands:

```bash
./main.py --help
```

To see options for a specific command:

```bash
./main.py <command> --help
```

## Data Processing

### Processing the 5G KPI Dataset

```bash
./main.py process-data --dataset 5g_kpi
```

Options:
- `--dataset`: Dataset to process (`5g_kpi`, `youtube`, or `all`)
- `--extract-features`: Run feature extraction after processing

### Processing All Datasets

```bash
./main.py process-data --dataset all
```

### Data Processing Pipeline

The data processing pipeline performs the following steps:

1. **Download**: Downloads the dataset if not already present
2. **Extract**: Extracts compressed files
3. **Clean**: Removes invalid entries and handles missing values
4. **Transform**: Converts data types and formats
5. **Feature Engineering**: Creates derived features
6. **Save**: Saves processed data to the `data/processed` directory

### Feature Extraction

Feature extraction creates windowed sequences for time-series prediction:

```bash
./main.py process-data --dataset 5g_kpi --extract-features
```

This process:
1. Loads processed data
2. Normalizes features
3. Creates sequences of past measurements
4. Pairs sequences with future values as targets
5. Splits data into training, validation, and test sets
6. Saves sequences to the `data/features` directory

## Model Training

### Training an LSTM Model

```bash
./main.py train --model lstm --dataset 5g_kpi
```

Options:
- `--model`: Model type (`lstm` or `transformer`)
- `--dataset`: Dataset to use (`5g_kpi`, `youtube`, or `all`)
- `--config`: Path to model configuration JSON file

### Custom Model Configuration

You can customize model hyperparameters using a JSON configuration file:

```bash
./main.py train --model lstm --config configs/lstm_config.json
```

Example configuration file (`configs/lstm_config.json`):

```json
{
  "lstm_units": 128,
  "dropout_rate": 0.3,
  "learning_rate": 0.001,
  "batch_size": 128,
  "epochs": 200,
  "patience": 20
}
```

### Training Process

The training process:

1. Loads the dataset
2. Builds the model architecture
3. Trains the model with early stopping
4. Evaluates the model on the test set
5. Saves the model and results

### Training Output

Training produces the following outputs in the `models` directory:

- **Saved Model**: `models/saved/<dataset>_<model>_model_<timestamp>.h5`
- **Model Architecture**: `models/saved/<dataset>_<model>_model_<timestamp>.json`
- **Training History Plot**: `models/results/<dataset>_<model>_history_<timestamp>.png`
- **Prediction Examples Plot**: `models/results/<dataset>_<model>_predictions_<timestamp>.png`
- **Evaluation Results**: `models/results/<dataset>_<model>_results_<timestamp>.txt`

## Model Evaluation

### Evaluating a Single Model

```bash
./main.py evaluate --models lstm
```

### Comparing Multiple Models

```bash
./main.py evaluate --models lstm transformer
```

Options:
- `--models`: Models to evaluate (`lstm`, `transformer`, or both)
- `--model-paths`: Path to JSON file with model paths

### Evaluation Metrics

The evaluation produces the following metrics for each target variable:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same units as the target
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **R² (R-squared)**: Proportion of variance explained by the model

### Visualization

Evaluation produces visualizations to help understand model performance:

- **Training History**: Loss and MAE over epochs
- **Prediction Examples**: Comparison of predictions with actual values
- **Feature Importance**: Analysis of which features contribute most to predictions

## Real-Time Prediction

### Running Real-Time Prediction

```bash
./main.py predict --model lstm --interface eth0
```

Options:
- `--model`: Model type (`lstm` or `transformer`)
- `--model-path`: Path to a specific model file
- `--interface`: Network interface to monitor
- `--buffer-size`: Number of seconds to keep in the buffer
- `--update-interval`: Seconds between updates

### Prediction Process

The real-time prediction process:

1. Loads the trained model
2. Monitors network metrics on the specified interface
3. Maintains a buffer of recent measurements
4. Makes predictions about future network conditions
5. Displays the predictions in real-time

### Output Format

The real-time predictor outputs:

- Current network metrics
- Predicted network metrics for the future
- Confidence intervals for predictions
- Visualization of trends

## Integration with Video Players

The system includes example code for integrating with video players:

```bash
./examples/video_player_integration.py
```

### Integration Components

The integration consists of three main components:

1. **QoS Predictor**: Makes predictions about future network conditions
2. **ABR Controller**: Decides on quality levels based on predictions
3. **Video Player**: Handles video playback and quality switching

### Integration Process

To integrate with your own video player:

1. Initialize the QoS predictor with a trained model
2. Create a network monitoring thread to collect measurements
3. Feed measurements to the predictor
4. Use predictions to make proactive quality decisions
5. Apply quality decisions to the video player

### Example Code

See `examples/video_player_integration.py` for a complete example of integration with a mock video player.

## Advanced Configuration

### Custom Datasets

To add a custom dataset:

1. Create a processing script in the `preprocessing` directory
2. Implement the required functions:
   - `download_dataset()`
   - `extract_dataset()`
   - `process_dataset()`
3. Update the `main.py` script to include the new dataset

### Custom Models

To add a custom model architecture:

1. Create a model building function in `models/train_model.py`
2. Implement the required components:
   - Model architecture
   - Compilation settings
   - Training procedure
3. Update the `main.py` script to include the new model type

### Hyperparameter Tuning

For hyperparameter tuning:

1. Create a configuration file with parameter ranges
2. Use the `--config` option with the training command
3. Analyze the results to find optimal parameters

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: Dependencies fail to install
**Solution**: Check Python version and ensure you have the required system libraries:

```bash
sudo apt-get install python3-dev libhdf5-dev
```

#### Data Processing Errors

**Issue**: Dataset download fails
**Solution**: Download the dataset manually and place it in the `data/raw` directory

#### Training Errors

**Issue**: Out of memory during training
**Solution**: Reduce batch size or sequence length in the configuration file

#### Prediction Errors

**Issue**: Model not found
**Solution**: Specify the model path explicitly with the `--model-path` option

### Logging

The system uses Python's logging module. To increase log verbosity:

```bash
export LOGLEVEL=DEBUG
./main.py <command> [options]
```

## FAQ

### General Questions

**Q: How accurate are the predictions?**
A: The model achieves R² scores of 0.72-0.92 for different metrics, with RTT being the most accurate.

**Q: How far into the future can the model predict?**
A: By default, the model predicts 3 seconds ahead, but this can be configured during feature extraction.

**Q: Can I use this on a mobile device?**
A: Yes, the trained models can be converted to TensorFlow Lite format for mobile deployment.

### Technical Questions

**Q: What deep learning framework is used?**
A: The system uses TensorFlow/Keras for model building and training.

**Q: How much data is needed for training?**
A: The system works best with at least several hours of network measurements.

**Q: Can I use GPU for training?**
A: Yes, if TensorFlow detects a compatible GPU, it will use it automatically.

---

For more information, please refer to the [User Guide](USER_GUIDE.md) and [Quick Start Guide](QUICK_START.md).