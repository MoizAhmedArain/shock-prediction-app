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

model = load_model("keras_model_v1.keras")

start = "2010-06-01"
ends = datetime.today() - timedelta(days=1)
end = ends.strftime('%Y-%m-%d')
df = yf.download("AAPL", start = start ,end= end )

st.title("Stock trend Prediction")

user_input = st.text_input("Enter Stock ticker",'AAPL')
df = yf.download(user_input, start = start ,end = end )

st.subheader("Date from 2010 - 2019")
st.write(df.describe())

st.subheader("Stock data")
st.write(df)

#visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, "b")
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100 days mean")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, "r")
plt.plot(df.Close,"b")
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100 & 200 days mean")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,"r")
plt.plot(ma200,"g")
plt.plot(df.Close, "b")
st.pyplot(fig)

# training / testing split (as you had)
data_training = pd.DataFrame(df["Close"][0:int(len(df) * 0.70)])
data_testing  = pd.DataFrame(df["Close"][int(len(df) * 0.70):])

# scale on training only
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)  # fit on train only

# build x_train, y_train using 100-day windows
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i, 0])   # <-- slice
    y_train.append(data_training_array[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))  # (n_samples, 100, 1)

# build model 
model = Sequential()

model.add(LSTM(60, return_sequences=True, input_shape=(x_input.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(80, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32)

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

y_predicted = model.predict(x_test)

y_predicted = scaler.inverse_transform(y_predicted)         
y_test = scaler.inverse_transform(y_test.reshape(-1,1))     

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
scaler = MinMaxScaler(feature_range=(0,1))
scaled100 = scaler.fit_transform(da100) 
x_input = np.array(scaled100).reshape(1, 100, 1) 
pred_scaled = model.predict(x_input)
pred_price = scaler.inverse_transform(pred_scaled)
actual_prices = df["Close"].tail(100)
all_prices = actual_prices + [pred_price[0][0]]

#plot- prediction
st.header("Next Day Prediction")
plt.figure(figsize=(12,6))
plt.plot(range(len(all_prices)), all_prices, "bo-", label="Actual + Prediction")
plt.axvline(x=len(all_prices)-1, color="r", linestyle="--", label="Prediction Point")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
st.pyplot(plt)



