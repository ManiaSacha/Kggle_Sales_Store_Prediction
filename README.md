# Store Sales Prediction - Kaggle Competition

This repository contains a PyTorch implementation for the Kaggle competition "Store Sales - Time Series Forecasting".

## Overview
The goal of this competition is to build a model that accurately predicts the unit sales for thousands of items sold at different Favorita stores in Ecuador.

## Project Architecture

### Data Processing Pipeline
- **Loading**: Loads various datasets (train, test, oil prices, stores, holidays, transactions)
- **Preprocessing**: 
  - Handles date features with proper datetime conversion
  - Pivots sales data by store and product family for time series format
  - Scales features using MinMaxScaler
  - Prepares sequences with sliding window approach for LSTM input

### LSTM Model Architecture
- **Input Layer**: Takes in sequences of store-family sales combinations
- **LSTM Layers**: 2 stacked LSTM layers with 200 hidden units each
- **Regularization**: 
  - Dropout (0.2) between LSTM layers
  - Batch normalization for improved training stability
- **Output Layer**: Dense layer that predicts sales for each store-family combination

### Training and Prediction Pipeline
- **Training Loop**: Uses early stopping with patience=100
- **Loss Function**: Custom RMSLE (Root Mean Squared Logarithmic Error) loss
- **Optimizer**: Adam with learning rate scheduling
- **Validation**: Uses a time-based split for validation
- **Prediction**: Generates predictions for test data and formats for Kaggle submission

## Project Structure
- `data_processing.py`: Functions for data loading and preprocessing
- `model.py`: PyTorch LSTM model implementation
- `train.py`: Training script
- `predict.py`: Script to generate predictions
- `utils.py`: Utility functions

## Requirements
Required packages are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

## Usage
1. Download the competition data from Kaggle and place it in a `data` folder
2. Run data preprocessing: `python data_processing.py`
3. Train the model: `python train.py`
4. Generate predictions: `python predict.py`

## Model Architecture
This implementation uses LSTM (Long Short-Term Memory) neural networks for time series forecasting. The model takes into account:
- Historical sales data
- Store and product family information
- Oil price data
- Holiday information

## Model Performance

### Training Results
- **Model**: LSTM with batch normalization and dropout
- **Training RMSLE**: ~0.054
- **Validation RMSLE**: ~0.09
- **Early Stopping**: Activated after ~197 epochs
- **Optimizer**: Adam with learning rate scheduler

### Prediction Results
- Successfully generated predictions for 28,512 test data points
- 96.6% of predictions have positive (non-zero) sales values
- The model successfully captures the sales patterns for different store-product combinations

## Future Improvements
- Incorporate oil price data as additional features
- Add holiday and events information
- Experiment with different sequence lengths
- Try more advanced architectures (GRU, Transformer)
- Implement ensemble methods combining multiple models
