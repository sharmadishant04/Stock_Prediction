# Stock Prediction using LSTM

This repository contains code for predicting stock prices of HDFC, IDFC, and KOTAK BANK using Long Short-Term Memory (LSTM) neural networks. The project includes data exploration, visualization, and model development to forecast future stock prices.

## Overview

The goal of this project is to predict the closing prices of stocks for HDFC, IDFC, and KOTAK BANK using historical data. The LSTM model is employed due to its capability to capture temporal dependencies in sequential data, which is suitable for time series forecasting like stock prices.

## Files

- **Stock_Prediction_using_LSTM.ipynb**: Jupyter Notebook containing the complete code for data loading, preprocessing, EDA, model training, and predictions.
- **HDFC BANK - Sheet1.csv**: CSV file containing historical stock data for HDFC BANK.
- **KOTAK - Sheet1.csv**: CSV file containing historical stock data for KOTAK BANK.
- **IDFC BANK - Sheet1.csv**: CSV file containing historical stock data for IDFC BANK.

## Components

### 1. EDA (Exploratory Data Analysis)

- Visualizing stock prices over time.
- Analyzing daily returns and volume traded.
- Calculating moving averages and correlations between stocks.

### 2. LSTM Model Development

- Data preprocessing and scaling.
- Creating sequences for LSTM training.
- Building and training the LSTM model for each stock.
- Evaluating model performance and predicting future stock prices.

## Requirements

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, keras (TensorFlow backend)

## Usage

To run the project and replicate the results, follow these steps:

### 1. Clone the repository

Clone this repository to your local machine using the following command:

```sh
git clone <repository_url>
cd <repository_name>
```


### 2. Install dependencies

Navigate to the project directory and install the required dependencies:

```sh
pip install -r requirements.txt
```


Make sure you have Python 3.x installed along with the necessary libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, and keras (TensorFlow backend).

### 3. Run the Jupyter Notebook

Open and execute the Jupyter Notebook `Stock_Prediction_using_LSTM.ipynb`:

```sh
jupyter notebook Stock_Prediction_using_LSTM.ipynb
```


### 4. Execute the notebook

Follow the instructions within the notebook to execute each cell sequentially. This includes:
- Loading and preprocessing historical stock data for HDFC, IDFC, and KOTAK BANK.
- Visualizing stock prices over time, daily returns, and volume traded.
- Building, training, and evaluating the LSTM model for each stock.
- Predicting future stock prices and comparing them with actual closing prices.

### 5. Review Results

Upon completion of the notebook execution, you will see:
- Visualizations depicting historical stock prices, daily returns, and trading volume.
- Predicted versus actual closing prices for HDFC, IDFC, and KOTAK BANK stocks.
- Evaluation metrics such as Root Mean Squared Error (RMSE) for model performance.

## Requirements

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, keras (TensorFlow backend)

## Files

- **Stock_Prediction_using_LSTM.ipynb**: Jupyter Notebook containing the complete code for data loading, preprocessing, EDA, model training, and predictions.
- **HDFC BANK - Sheet1.csv**: CSV file containing historical stock data for HDFC BANK.
- **KOTAK - Sheet1.csv**: CSV file containing historical stock data for KOTAK BANK.
- **IDFC BANK - Sheet1.csv**: CSV file containing historical stock data for IDFC BANK.

## Components

### 1. EDA (Exploratory Data Analysis)

- Visualizing stock prices over time.
- Analyzing daily returns and volume traded.
- Calculating moving averages and correlations between stocks.

### 2. LSTM Model Development

- Data preprocessing and scaling.
- Creating sequences for LSTM training.
- Building and training the LSTM model for each stock.
- Evaluating model performance and predicting future stock prices.

## Results

This project demonstrates how LSTM models can be utilized for predicting stock prices based on historical data. It provides insights into the potential future performance of HDFC, IDFC, and KOTAK BANK stocks, helping investors in making informed decisions.

## Conclusion

By following these steps, you can replicate the experiment and explore further enhancements or modifications to the model for better predictions.
