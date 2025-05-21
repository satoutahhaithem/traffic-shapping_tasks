# AI QoS Predictor (Lightweight Package)

This is a lightweight version of the AI QoS Predictor for Mobile Video Streaming project, packaged for easy deployment to another device.

## What's Included

- Core Python scripts for prediction and model training
- Pre-trained LSTM and Transformer models
- Configuration files
- Essential documentation
- Setup script

## Directory Structure

```
ai_qos_predictor_lightweight/
├── ai_qos_predictor/           # Main project directory
│   ├── data/                   # Data directories (empty)
│   ├── evaluation/             # Evaluation directory (empty)
│   ├── models/                 # Model definitions
│   ├── preprocessing/          # Preprocessing scripts
│   ├── sample_configs/         # Model configurations
│   ├── main.py                 # Main entry point
│   ├── real_time_predictor.py  # Real-time prediction system
│   ├── README.md               # Project documentation
│   ├── requirements.txt        # Python dependencies
│   ├── setup.sh                # Setup script
│   └── QUICK_START.md          # Quick start guide
└── models/                     # Pre-trained models
    ├── best_model.pt           # Best LSTM model
    ├── best_transformer_model.pt # Best Transformer model
    ├── 5g_kpi_lstm_20250515_104704_config.json        # LSTM model config
    └── 5g_kpi_transformer_20250515_110105_config.json # Transformer model config
```

## Getting Started

1. Extract this package on the target device
2. Navigate to the ai_qos_predictor directory
3. Run the setup script:
   ```bash
   cd ai_qos_predictor
   chmod +x setup.sh
   ./setup.sh
   ```
4. Run the predictor:
   ```bash
   ./main.py predict --model lstm --interface eth0
   ```

## Notes

- This package includes pre-trained models, so you can start making predictions immediately
- The models directory is at the same level as the ai_qos_predictor directory to match the expected path in the code
- For full functionality, refer to the included README.md and QUICK_START.md files