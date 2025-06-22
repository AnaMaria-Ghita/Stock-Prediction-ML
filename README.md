# Stock Price Prediction using LSTM

## Overview

This project predicts the next-day stock price for a given ticker (default: BKNG) using historical data and an LSTM neural network. It covers data collection, preprocessing, feature engineering, model training, evaluation, and visualization.

## Features

- Download stock data with `yfinance`
- Calculate moving averages, daily returns, and volatility
- Prepare sequences for LSTM input
- Train an LSTM model using TensorFlow/Keras
- Evaluate model performance with RMSE and MAE
- Visualize actual vs predicted prices
- Predict next-day adjusted closing price
