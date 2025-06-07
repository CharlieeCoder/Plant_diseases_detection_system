# Plant Diseases Detection System

A deep learning-based system for detecting diseases in plants using image classification.

## Features

- Plant disease detection using CNN architecture
- Real-time image processing and classification
- Support for multiple plant types and diseases
- Training pipeline with data augmentation
- Evaluation metrics and visualization
- Model checkpointing and training history tracking

## Project Structure

```
├── train/              # Training dataset
├── valid/              # Validation dataset
├── test/               # Test dataset
├── static/             # Static files for web interface
├── view/               # View templates
├── debug_images/       # Debug visualization
├── retrain.py          # Model training script
├── main.py            # Main application script
├── evaluate_metrics.py # Evaluation script
├── update_plots.py    # Visualization update script
└── requirement.txt    # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CharlieeCoder/Plant_diseases_detection_system.git
cd Plant_diseases_detection_system
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

## Usage

1. Training the model:
```bash
python retrain.py
```

2. Running the application:
```bash
python main.py
```

3. Evaluating the model:
```bash
python evaluate_metrics.py
```

## Model Architecture

The system uses a CNN architecture with:
- Multiple convolutional layers with batch normalization
- MaxPooling layers
- Dropout for regularization
- Dense layers for classification

## Performance Metrics

- Accuracy
- Top-3 Accuracy
- AUC Score
- Loss metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 