# AI QoS Predictor: Quick Start Guide

This guide provides step-by-step instructions to get started with the AI QoS Predictor system.

## 1. Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/ai_qos_predictor.git
cd ai_qos_predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Data Processing

### Process the 5G KPI Dataset

```bash
./main.py process-data --dataset 5g_kpi
```

This command will:
1. Download the 5G KPI dataset (if not already present)
2. Extract and process the CSV files
3. Create normalized features for model training

Expected output:
```
INFO - Processing 5g_kpi dataset
INFO - Dataset already exists at /path/to/ai_qos_predictor/data/raw/5g_kpi/5g_dataset.zip
INFO - Extracting dataset...
INFO - Processing CSV files...
INFO - Total records: 188711
INFO - 5g_kpi dataset processed successfully
```

## 3. Model Training

### Train the LSTM Model

```bash
./main.py train --model lstm --dataset 5g_kpi
```

This command will:
1. Load the processed 5G KPI dataset
2. Create an LSTM model for time-series prediction
3. Train the model with early stopping
4. Evaluate the model on the test set
5. Save the trained model and results

Expected output:
```
INFO - Training lstm model
INFO - Loading data for 5g_kpi
INFO - Training samples: 113219
INFO - Validation samples: 37740
INFO - Test samples: 37740
INFO - Model built
INFO - Training model...
Epoch 1/100
1770/1770 [==============================] - 14s 8ms/step - loss: 0.3933 - mae: 0.1900 - val_loss: 0.2746 - val_mae: 0.1133
...
INFO - Model training completed
INFO - Evaluating model...
INFO - Metrics for throughput_down:
INFO -   r2: 0.7181
INFO - Metrics for rtt:
INFO -   r2: 0.9234
INFO - Metrics for packet_loss:
INFO -   r2: 0.7676
INFO - Model saved to /path/to/ai_qos_predictor/models/saved/5g_kpi_lstm_model_TIMESTAMP.h5
```

## 4. Making Predictions

### Real-time Prediction

```bash
./main.py predict --model lstm --interface eth0
```

This command will:
1. Load the trained model
2. Monitor network metrics on the specified interface
3. Make predictions about future network conditions
4. Display the predictions in real-time

## 5. Viewing Results

### Training History

The training history plot shows the loss and MAE over epochs:

```bash
open models/results/5g_kpi_lstm_history_*.png
```

### Prediction Examples

The prediction plot shows examples of the model's predictions compared to actual values:

```bash
open models/results/5g_kpi_lstm_predictions_*.png
```

### Detailed Results

The results text file contains detailed metrics and model information:

```bash
cat models/results/5g_kpi_lstm_results_*.txt
```

## 6. Next Steps

1. Try different model architectures:
   ```bash
   ./main.py train --model transformer --dataset 5g_kpi
   ```

2. Process additional datasets:
   ```bash
   ./main.py process-data --dataset youtube
   ```

3. Compare multiple models:
   ```bash
   ./main.py evaluate --models lstm transformer
   ```

For more detailed information, refer to the [User Guide](USER_GUIDE.md).