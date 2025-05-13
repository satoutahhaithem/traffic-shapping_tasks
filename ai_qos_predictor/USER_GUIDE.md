# AI QoS Predictor for Mobile Video Streaming: A Complete Guide

This document provides a comprehensive explanation of the AI QoS Predictor project, designed for readers with no prior knowledge of the system. It covers the datasets used, code structure, implementation details, and results.

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Datasets](#datasets)
4. [Project Structure](#project-structure)
5. [Implementation Details](#implementation-details)
6. [Results and Evaluation](#results-and-evaluation)
7. [How to Use the System](#how-to-use-the-system)
8. [Future Work](#future-work)

## Introduction

The AI QoS Predictor is a machine learning system designed to predict future network conditions for mobile video streaming. By analyzing patterns in network metrics like throughput, round-trip time (RTT), and packet loss, the system can anticipate network degradation before it affects video quality, allowing streaming applications to take proactive measures to maintain a good user experience.

## Problem Statement

Mobile video streaming quality is highly dependent on network conditions, which can fluctuate rapidly, especially in moving vehicles. When network conditions deteriorate, video playback may stall, quality may drop, or buffering may occur, leading to a poor user experience.

Traditional adaptive streaming approaches react to network changes after they occur, resulting in a delay between network degradation and quality adaptation. Our goal is to predict network changes before they happen, allowing streaming applications to adapt proactively.

## Datasets

### 5G KPI Dataset

The primary dataset used in this project is the **5G KPI Dataset** (also known as "Beyond Throughput"), which contains network measurements from mobile devices in both stationary and moving vehicles. This dataset is particularly valuable because it captures real-world network fluctuations in mobile scenarios.

#### Dataset Details:
- **Source**: ACM Digital Library, "Beyond throughput, the next generation: a 5G dataset with channel and context metrics"
- **Size**: Approximately 180 MB (compressed)
- **Format**: CSV files organized by service type (Amazon Prime, Netflix, Download) and mobility (Driving, Static)
- **Time Period**: Data collected in 2019-2020
- **Location**: Ireland

#### Key Features:
- **Throughput (downlink/uplink)**: The data transfer rate in bits per second
- **RTT (Round-Trip Time)**: The time it takes for a packet to travel from source to destination and back
- **Jitter**: Variation in packet delay
- **Packet Loss**: Percentage of packets that fail to reach their destination
- **Signal Quality Metrics**: SINR (Signal-to-Interference-plus-Noise Ratio), RSRP (Reference Signal Received Power), RSRQ (Reference Signal Received Quality)
- **Contextual Information**: GPS coordinates, speed, cell ID

### Other Datasets (Planned)

While not yet implemented, the system is designed to incorporate additional datasets:

- **YouTube Mobile Streaming Dataset**: Contains video QoE (Quality of Experience) metrics like stall events, quality levels, and buffer health
- **Simula HSDPA 3G Drive Traces**: Legacy network data with extreme bad-coverage events
- **Kaggle Adaptive Bandwidth Dataset**: Simple dataset for rapid prototyping

## Project Structure

The project is organized into several components:

```
ai_qos_predictor/
├── data/                      # Data storage
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed datasets
│   └── features/              # Extracted features for ML
├── preprocessing/             # Data preprocessing scripts
│   ├── process_5g_dataset.py  # 5G KPI dataset processor
│   └── feature_extraction.py  # Feature extraction for ML
├── models/                    # ML model implementations
│   ├── train_model.py         # Model training script
│   ├── saved/                 # Saved model files
│   └── results/               # Training results and plots
├── evaluation/                # Model evaluation tools
├── main.py                    # Main entry point
└── README.md                  # Project documentation
```

## Implementation Details

### Data Processing Pipeline

The data processing pipeline consists of several steps:

1. **Data Download and Extraction**: The raw 5G KPI dataset is downloaded and extracted.
2. **Data Cleaning**: Missing values are handled, and data types are converted appropriately.
3. **Feature Engineering**: Raw metrics are processed to create useful features for the model.
4. **Normalization**: Features are normalized to have zero mean and unit variance.
5. **Sequence Creation**: Time-series data is converted into windowed sequences for prediction.

#### Code Example: Data Processing

```python
# Load and process the 5G KPI dataset
def process_5g_dataset():
    # Download dataset if not already present
    download_dataset()
    
    # Process each CSV file
    for csv_file in find_csv_files():
        df = pd.read_csv(csv_file)
        
        # Clean data
        df = clean_data(df)
        
        # Extract features
        df = extract_features(df)
        
        # Save processed data
        save_processed_data(df)
```

### Feature Extraction

Feature extraction transforms the processed data into a format suitable for machine learning:

1. **Sequence Creation**: For each time point, a sequence of the previous 10 seconds of data is used as input.
2. **Target Selection**: The network conditions 3 seconds in the future are used as targets.
3. **Feature Alignment**: Features are aligned across different datasets.
4. **Normalization**: Features are normalized using StandardScaler.
5. **Train/Validation/Test Split**: Data is split into training (60%), validation (20%), and test (20%) sets.

#### Code Example: Feature Extraction

```python
def create_sequences(df, sequence_length, prediction_horizon, features, targets):
    """
    Create windowed sequences for time-series prediction
    """
    X, y = [], []
    
    for i in range(len(df) - sequence_length - prediction_horizon + 1):
        # Input sequence: current window of features
        X.append(df[features].values[i:i+sequence_length])
        
        # Target: future values of target features
        y.append(df[targets].values[i+sequence_length+prediction_horizon-1])
    
    return np.array(X), np.array(y)
```

### Model Architecture

The primary model used is an LSTM (Long Short-Term Memory) neural network, which is well-suited for time-series prediction:

1. **Input Layer**: Takes sequences of shape (10, 8) - 10 time steps with 8 features each.
2. **First LSTM Layer**: 64 units with return_sequences=True to maintain the time dimension.
3. **Dropout Layer**: 20% dropout for regularization.
4. **Second LSTM Layer**: 64 units.
5. **Dropout Layer**: Another 20% dropout.
6. **Output Layer**: Dense layer with 3 units (one for each target metric).

#### Code Example: Model Architecture

```python
def build_lstm_model(input_shape, output_shape, lstm_units=64, dropout_rate=0.2):
    """
    Build an LSTM model for time-series prediction
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(output_shape)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

### Training Process

The model is trained using the following approach:

1. **Optimizer**: Adam optimizer with default learning rate.
2. **Loss Function**: Mean Squared Error (MSE).
3. **Metrics**: Mean Absolute Error (MAE).
4. **Early Stopping**: Training stops when validation loss doesn't improve for 10 epochs.
5. **Learning Rate Reduction**: Learning rate is reduced when validation loss plateaus.
6. **Model Checkpointing**: The best model based on validation loss is saved.

#### Code Example: Model Training

```python
def train_model(model, data, epochs=100, batch_size=64, patience=10):
    """
    Train the model with early stopping
    """
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        data['X_train'],
        data['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data['X_val'], data['y_val']),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history
```

## Results and Evaluation

### Training Results

The LSTM model was trained on 113,219 sequences from the 5G KPI dataset, with 37,740 sequences each for validation and testing. After training for 46 epochs with early stopping, the model achieved the following results:

#### Final Training Metrics:
- **Training Loss (MSE)**: 0.1284
- **Training MAE**: 0.1060
- **Validation Loss (MSE)**: 0.2042
- **Validation MAE**: 0.0894

### Test Set Evaluation

The model was evaluated on the test set, with the following results for each target metric:

| Target Metric | MSE | RMSE | MAE | R² |
|---------------|-----|------|-----|-----|
| throughput_down | 0.2822 | 0.5313 | 0.1685 | 0.7181 |
| rtt | 0.0758 | 0.2753 | 0.0456 | 0.9234 |
| packet_loss | 0.2128 | 0.4613 | 0.0519 | 0.7676 |

These results indicate that:
- The model predicts RTT with high accuracy (R² = 0.92)
- Throughput and packet loss predictions are good but less accurate (R² = 0.72 and 0.77, respectively)
- The low MAE values suggest that the model's predictions are close to the actual values

### Visualization

The training process and predictions are visualized through plots:

1. **Training History**: Shows the loss and MAE over epochs for both training and validation sets.
2. **Predictions**: Compares the model's predictions with actual values for a few test samples.

## How to Use the System

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, Scikit-learn

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai_qos_predictor.git
   cd ai_qos_predictor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data Processing

To process the 5G KPI dataset:

```
./main.py process-data --dataset 5g_kpi
```

### Model Training

To train the LSTM model:

```
./main.py train --model lstm --dataset 5g_kpi
```

### Model Evaluation

To evaluate the trained model:

```
./main.py evaluate --models lstm
```

### Making Predictions

To use the trained model for real-time predictions:

```
./main.py predict --model lstm --interface eth0
```

## Future Work

1. **Additional Datasets**: Incorporate the YouTube mobile dataset to add QoE metrics.
2. **Alternative Models**: Implement and compare Transformer-based models.
3. **Real-time Integration**: Develop a real-time prediction system that integrates with video players.
4. **Web Interface**: Create a web dashboard for visualizing predictions and network conditions.
5. **Mobile App**: Develop a mobile app that uses the model for on-device predictions.

---

This document provides a comprehensive overview of the AI QoS Predictor project. For more detailed information, please refer to the code documentation and comments within the source files.