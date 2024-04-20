import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf

yf.pdr_override()
from keras.models import load_model
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import requests
import numpy as np


def calculate_smape(actual, predicted):
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)

    return round(
        np.mean(
            np.abs(predicted - actual) /
            ((np.abs(predicted) + np.abs(actual)) / 2)
        ) * 100, 2
    )


startdate = datetime(2013, 1, 1)
enddate = datetime(2023, 12, 31)
f = open("style.css")
string_css = f.read()
st.markdown(
    f"""
    <style>
    {string_css}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Stock Trend Prediction')


class SessionState:
    def __init__(self):
        self.select_symbols = 'AAPL'


# Get all list NASDAQ
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
res = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers)
df_profile = pd.DataFrame(res.json()['data']['data']['rows'])
list_symbols = df_profile['symbol'].to_list()

# Tạo một select box
session_state = SessionState()

col1, col2 = st.columns(3)[0:2]
with col1:
    select_symbols = st.selectbox('Select stock symbols', options=sorted(list_symbols), index=0)
# with col2:
#     select_date = st.selectbox('Select intevals to predict (days)', options=[i for i in range(1, 11)], index=0)
session_state.select_symbols = select_symbols
# session_state.select_date = select_date


# user_input = st.text_input('Enter Stock sticker', 'AAPL')

def write_to_screen(select_symbols):
    df = pdr.get_data_yahoo(select_symbols, start=startdate, end=enddate)

    profile = df_profile[df_profile['symbol'] == select_symbols].reset_index()

    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader('General Infomation')
        for col in profile.columns:
            info = profile.loc[0, col]
            st.write(
                f'<div style="display:flex"><p style ="font-weight:bold; text-transform: uppercase">{col.lower()}: </p>&nbsp&nbsp&nbsp <p>{info}</p></div>',
                unsafe_allow_html=True)
    # Describing Data
    with col2:
        st.subheader('Data from 2013- 2023')
        st.write(df.describe())

    # Visualization
    st.subheader('I. Data Visualization')
    colx, coly = st.columns(2)
    with colx:
        st.text('1. Closing Price vs Time chart')
        plot = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'])
        st.pyplot(plot)

    with coly:
        st.text('2. Closing price with Rolling100 and Rolling200')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        plot1 = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], 'b', label='Original Price')
        plt.plot(ma100, 'r', label='Rolling 100')
        plt.plot(ma200, 'g', label='Rolling 200')
        plt.legend()
        plt.show()
        st.pyplot(plot1)

    # Splitting data into Training and Testing
    data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    data_test = pd.DataFrame(df['Close'][int(len(df) * 0.7):len(df)])

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_array = scaler.fit_transform(data_train)

    # Load my model
    model = load_model('stock_pre.h5')

    # Testing_part
    with st.spinner('Model is running'):
        past_100_days = data_train.tail()

        final_test = pd.concat([past_100_days, data_test], ignore_index=True)

        input_data = scaler.fit_transform(final_test)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predicted = model.predict(x_test)

        scale = scaler.scale_

        scaler_factor = 1 / scale[0]
        y_predicted_1 = y_predicted * scaler_factor
        y_test_1 = y_test * scaler_factor

    # Metrics
    #metrics = ['MAE', 'MAPE', 'MSE', 'R2_Score', 'SMAPE']
    metrics = ['MAE', 'MSE', 'R2_Score']
    value = [mean_absolute_error(y_test_1,y_predicted_1),
            # mean_absolute_percentage_error(y_test_1,y_predicted_1),
             mean_squared_error(y_test_1,y_predicted_1),
             r2_score(y_test_1,y_predicted_1),
             #calculate_smape(y_test_1,y_predicted_1)
             ]

    st.subheader('II. RESULT OF MODEL')
    col3, col4 = st.columns([1,2])
    with col3:
        st.text('1. Evaluate Model')
        for i in range(len(metrics)):
            st.write(
                f'<div style="display:flex"><p style ="font-weight:bold; text-transform: uppercase">{metrics[i]}: </p>&nbsp&nbsp&nbsp <p>{value[i]}</p></div>',
                unsafe_allow_html=True)

    # Prediction graph
    with col4:
        st.text('2. Original Closing Price vs Predicted Closing Price')
        plot2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test_1, 'b', label='Original Price')
        plt.plot(y_predicted_1, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        st.pyplot(plot2)

    # # MAE
    # st.subheader('MAE')
    # mae = mean_absolute_error(y_test_1, y_predicted_1)
    # st.text(mae)


if st.button('Search'):
    write_to_screen(select_symbols)
