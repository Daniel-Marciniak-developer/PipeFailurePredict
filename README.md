# PipeFailurePredict--WaterPrime

## Overview
PipeFailurePredict--WaterPrime is a comprehensive solution for predicting water pipe failures in municipal water supply networks. The system combines hydraulic flow simulation with machine learning models to predict when and where pipe failures are likely to occur, enabling proactive maintenance and reducing service disruptions.

## Features
- **Flow Simulation**: Simulates water flow through pipe networks using the Hardy-Cross method
- **Failure Prediction**: Uses machine learning models to predict pipe failures:
  - Regression Model: Predicts days until failure
  - Transformer Model: Advanced time-series prediction of failure events
- **Visualization Tools**: Interactive dashboards for analyzing network topology and failure predictions
- **Data Generation**: Tools for generating synthetic data for testing and development

## Project Structure
```
PipeFailurePredict--WaterPrime/
├── FlowAlgorithm/               # Water flow simulation components
│   ├── data/                    # Input data for flow simulation
│   ├── output/                  # Simulation results
│   ├── GenerateDatabase/        # Tools for generating synthetic network data
│   ├── data_loader.py           # Data loading utilities
│   ├── flow_calculator.py       # Flow calculation algorithms
│   ├── flow_simulation.py       # Main simulation coordinator
│   ├── hydraulic_network.py     # Network graph representation
│   ├── main.py                  # Entry point for flow simulation
│   └── visualize_network.py     # Network visualization tools
│
├── Predictions/                 # Failure prediction components
│   ├── data/                    # Failure data for training and prediction
│   ├── GenerateDatabase/        # Tools for generating synthetic failure data
│   ├── OldPredictionModels/     # Previous model implementations
│   ├── RegressionModel/         # Neural network regression model
│   │   ├── models/              # Trained model files
│   │   ├── app.py               # Streamlit web application
│   │   └── model_training.py    # Model training script
│   └── TransformerModel/        # Transformer-based prediction model
│       ├── models/              # Trained model files
│       ├── train_model.py       # Model training script
│       └── visualization_app.py # Streamlit visualization app
│
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore file
├── Dockerfile                   # Docker container definition
├── docker-compose.yml           # Multi-container Docker setup
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA-compatible GPU (optional, for faster calculations)

### Option 1: Using Docker (Recommended)
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PipeFailurePredict--WaterPrime.git
   cd PipeFailurePredict--WaterPrime
   ```

2. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

3. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

### Option 2: Manual Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PipeFailurePredict--WaterPrime.git
   cd PipeFailurePredict--WaterPrime
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Running Flow Simulation
```bash
python -m FlowAlgorithm.main
```

### Visualizing Network Topology
```bash
python -m FlowAlgorithm.visualize_network
```

### Running Prediction Web Applications
```bash
# Regression Model App
streamlit run Predictions/RegressionModel/app.py

# Transformer Model Visualization App
streamlit run Predictions/TransformerModel/visualization_app.py
```

### Generating Synthetic Data
```bash
# Generate pipe network data
python -m FlowAlgorithm.GenerateDatabase.generate_database_pipe

# Generate water failure data
python -m Predictions.GenerateDatabase.generate_database_water_network
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The Hardy-Cross method for hydraulic network analysis
- PyTorch for machine learning capabilities
- NetworkX for graph-based network representation