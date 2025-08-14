# shock-prediction-app
# Stock Price Prediction App 

This project is a **Stock Price Prediction** tool built with **LSTM (Long Short-Term Memory) neural networks** in TensorFlow/Keras, and deployed using **Streamlit**.  
It allows users to input a stock ticker (e.g., AAPL, TSLA) and view both the **historical trend** and the **predicted price movement**.

Features
- Fetches historical stock data from Yahoo Finance.
- Preprocesses and scales the data for LSTM training.
- Predicts future price trends using a deep learning model.
- Displays interactive visualizations of **Actual vs Predicted Prices**.
- Simple web interface built with Streamlit.

 How It Works
1. **Data Fetching** → The app downloads historical data for the selected stock ticker.
2. **Data Preparation** → Data is scaled and transformed into sequences for the LSTM model.
3. **Prediction** → The trained LSTM model predicts prices based on the latest available data.
4. **Visualization** → Actual and predicted prices are plotted on a graph.

Important Note
When you run the app for the first time or change the stock ticker,  
**it can take 2 to3 minutes** to load the data and generate the prediction graph.  
This is due to the model loading and processing the input data.

Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt

