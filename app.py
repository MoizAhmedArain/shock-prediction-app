import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import streamlit as st
from keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

try:
    model = load_model("keras_model_v1.keras")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

start = "2010-06-01"
ends = datetime.today() #- timedelta(days=1)
end = ends.strftime('%Y-%m-%d')

st.title("Stock trend Prediction")
user_input = st.text_input("Enter Stock ticker",'AAPL')
try:
    df = yf.download(user_input, start = start ,end = end )
    if df.empty:
        raise ValueError("No data returned for ticker.")
except Exception as e:
    st.error(f"Data download failed: {e}")
    st.stop()

st.subheader("Date from 2010 to yesterday")
st.write(df.describe())

st.subheader("Stock data")
st.write(df)

#visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, "b")
plt.legend()
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100 days mean")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, "r")
plt.plot(df.Close,"b")
plt.legend()
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100 & 200 days mean")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,"r")
plt.plot(ma200,"g")
plt.plot(df.Close, "b")
plt.legend()
st.pyplot(fig)

# training / testing split 
data_training = pd.DataFrame(df["Close"][0:int(len(df) * 0.70)])
data_testing  = pd.DataFrame(df["Close"][int(len(df) * 0.70):])

# scale on training only
try:
    scaler = MinMaxScaler(feature_range=(0,1))
except Exception as e:
    st.error(f"Scaler Can't be perfromed: {e}")
    st.stop()
    
data_training_array = scaler.fit_transform(data_training) 

# build x_train, y_train using 100 day 
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i, 0])   
    y_train.append(data_training_array[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))  

# prepare test input (last 100 days of train + testing)
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.transform(final_df)  

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i, 0])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test).reshape((len(x_test), 100, 1))
y_test = np.array(y_test)

try:
    y_predicted = model.predict(x_test)
except Exception as e:
    st.error(f"model is not loading: {e}")
    st.stop()
    
y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
# plot
st.header("Stock Prediction vs Actual Data")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y_test, "b", label="Original Price")
ax.plot(y_predicted, "r", label="Predicted Price")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

#Prediction from 100 days
da100 = df['Close'].tail(100)
#scaler = MinMaxScaler(feature_range=(0,1))
scaled100 = scaler.fit_transform(da100) 
x_input = np.array(scaled100).reshape(1, 100, 1) 
try:
    pred_scaled = model.predict(x_input)
except Exception as e:
    st.error(f"model is not performing: {e}")
    st.stop()
pred_price = scaler.inverse_transform(pred_scaled)
actual_prices = df["Close"].tail(100)
all_prices = actual_prices + [pred_price[0][0]]

#plot- prediction
st.header("Next Day Prediction")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(all_prices)), all_prices, "bo-", label="Actual + Prediction")
ax.axvline(x=len(all_prices)-1, color="r", linestyle="--", label="Prediction Point")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)





