# Importing the required libraries
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

from datetime import date
import time as t
import yfinance as yf
from keras.models import load_model

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM

#from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
#from tensorflow.keras.optimizers import Adam

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

selected = option_menu(
    menu_title=None,
    options=["Prediction and Analysis", "Know More"],
    icons=["house", "book"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if selected == "Prediction and Analysis":

    st.title('Stock Forecast and Analysis App')

    st.sidebar.subheader("Page Navigation")
    analysis = st.sidebar.radio("Navigate to predictions page or analysis pages",
                                ('Home', 'Get Predictions', 'LSTM Analysis', 'CNN Analysis', 'SVM Analysis'))

    st.sidebar.subheader("Know Company Ticker")
    add_selectbox = st.sidebar.selectbox(
        "Enter company name for ticker value",
        ("Amazon: AMZN", "Apple Inc: AAPL", "ACC Limited: ACC.NS", "Adani Green Energy Limited: ADANIGREEN.NS", "Aditya Birla Capital Limited: ABCAPITAL.NS",
            "Asian Paints Limited: ASIANPAINT.NS", "Axis Bank Limited: AXISBANK.NS",
            "Bajaj Finance Limited: BAJFINANCE.NS", "Bharti Airtel Limited: BHARTIARTL.NS", "Britannia Industries Limited: BRITANNIA.NS",
            "Coal India Limited: COALINDIA.NS",
            "Global Payments Inc: GPN", "Google: GOOG",
            "HCL Technologies Limited: HCLTECH.NS", "HDFC Bank Limited: HDB", "Hindustan Unilever Limited: HINDUNILVR.NS",
            "ICICI Bank Limited: IBN", "Indian Oil Corporation Limited: IOC.NS", "Infosys Limited: INFY.NS",
            "JPMorgan Chase & Co: JPM", "JSW Steel Limited: JSWSTEEL.NS",
            "Kotak Mahindra Bank Limited: KOTAKBANK.NS",
            "Larsen & Toubro Limited: LT.NS", "Life Insurance Corporation of India: LICI.NS",
            "Mahindra & Mahindra Limited: M&M.NS", "Maruti Suzuki India Limited: MARUTI.NS", "Minerva Neurosciences, Inc: NERV", "Meta: META",
            "NVIDIA Corporation: NVDA", "Netflix Inc: NFLX", "NTPC Limited: NTPC.NS",
            "Oil and Natural Gas Corporation Limited: ONGC.NS",
            "Power Grid Corporation of India Limited: POWERGRID.NS",
            "Reliance Industries Limited: RELIANCE.NS",
            "SBI: SBIN.NS", "Starbucks Corporation: SBUX", "SoFi Technologies, Inc: SOFI",
            "TCS Limited: TCS.NS", "Titan Company Limited: TITAN.NS", "Tata Motors Limited: TATAMOTORS.NS",
            "Uber Technologies Inc: UBER", "UltraTech Cement Limited: ULTRACEMCO.NS",
            "Vedanta Limited: VEDL.NS", "Wipro Limited: WIT")
    )
    st.sidebar.write('Ticker of', add_selectbox)

    if analysis == 'Home':
        image1 = Image.open('images/stock.jpg')
        st.image(image1)

        st.subheader(
            'Welcome to the stock price prediction and visualization web app')

        st.write('Stock Price Prediction using machine learning helps you discover the future value of company stock and other financial assets traded on an exchange using the stock ticker. The entire idea of predicting stock prices is to gain significant profits.')

    if analysis == 'Get Predictions':

        stock = st.text_input('Enter Stock Ticker For Prediction', 'AAPL')
        st.write('Performing Analysis of ', stock)
        selected_stock = stock

        START = "2015-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        @st.cache_resource
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

        data_load_state = st.text('Loading data...')
        try:
            data = load_data(selected_stock)
            df = data
            t.sleep(2)
            data_load_state.text('Loading data... done!')
        except:
            st.write('Enter a valid ticker')
            data_load_state.text('Loading data... Error!')

        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data

        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(
                x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(
                title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        if st.button('Generate Predictions'):

            forecast_state = st.subheader('Generating Predictions...')
            t.sleep(2)
            forecast_state.subheader('Generating Predictions... done!')

            st.subheader("Showing 100 days and 200 days Moving Average")
            df = df.reset_index()
            # Moving average 100 calculation
            ma100 = df.Close.rolling(100).mean()
            # Moving average 200 calculation
            ma200 = df.Close.rolling(200).mean()

            # Chart for ma100 and ma200
            fig1 = plt.figure(figsize=(12, 6))
            plt.plot(df.Close)
            plt.plot(ma100, 'r')
            plt.plot(ma200, 'g')
            st.pyplot(fig1)
            st.write(df.shape)
            st.write('If 100 days MA crosses 200 days MA from above, it means bearish reversal may take place and if 100 days MA crosses 200 days MA from below, it means bullish reversal may take place.')

            # Predict forecast with Prophet.
            df_train = data[['Date', 'Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            # Show and plot forecast
            st.subheader('Forecast data')
            st.write(forecast.tail())

            st.subheader(f'Forecast plot for {n_years} years')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.subheader("Forecast weekly, monthly and yearly components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)

        else:
            st.write('Click on Generate Predictions to get future stock values.')

    if analysis == 'LSTM Analysis':

        st.subheader("Using LSTM Model")

        stock = st.text_input('Enter Stock Ticker For Prediction', 'SBIN.NS')
        st.write('Performing Analysis of ', stock)
        selected_stock = stock
        Start = "2015-01-01"
        Today = date.today().strftime("%Y-%m-%d")

        data_load_state = st.text('Loading data...')
        try:
            data = yf.download(selected_stock, Start, Today)
            t.sleep(2)
            data_load_state.text('Loading data... Done!')
        except:
            st.write('Enter a valid ticker')
            data_load_state.text('Loading data... Error!')

        df = data
        st.subheader('Raw Data Description')
        st.write(data.tail())

        # Preprocessing the data
        # Resample to daily frequency, forward fill missing values, and drop any remaining NaNs
        data = data['Close'].resample('D').ffill().dropna()

        # Scaling the data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(np.array(data).reshape(-1, 1))

        # Creating the features and target variables
        lookback = 60  # Number of days to look back
        X = []
        y = []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)

        # Splitting the data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # Reshaping the data for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        st.subheader("Showing 100 days and 200 days Moving Average")
        df = df.reset_index()
        # Moving average 100 calculation
        ma100 = df.Close.rolling(100).mean()
        # Moving average 200 calculation
        ma200 = df.Close.rolling(200).mean()

        # Chart for ma100 and ma200
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df.Close)
        plt.plot(ma100, 'r')
        plt.plot(ma200, 'g')
        st.pyplot(fig1)
        st.write(df.shape)
        st.write('If 100 days MA crosses 200 days MA from above, it means bearish reversal may take place and if 100 days MA crosses 200 days MA from below, it means bullish reversal may take place.')

        # load Model
        model = load_model('keras_LSTM_model.h5')

        # Predicting the stock prices for the test set
        predicted_stock_prices = model.predict(X_test)
        predicted_stock_prices = scaler.inverse_transform(
            predicted_stock_prices)

        if st.button('Visualize Predictions'):
            # Visualizing the results
            st.subheader('Visualizing the Predicted Stock Prices Using LSTM')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(scaler.inverse_transform(
                y_test.reshape(-1, 1)), label='Actual Stock Price')
            plt.plot(predicted_stock_prices, label='Predicted Stock Price')
            plt.legend()
            st.pyplot(fig2)
        else:
            st.write('Click on Visualize Predictions to get the further analysis.')

    if analysis == 'CNN Analysis':

        st.subheader("Using CNN Model")

        stock = st.text_input('Enter Stock Ticker For Prediction', 'GOOG')
        st.write('Performing Analysis of ', stock)
        selected_stock = stock
        Start = "2015-01-01"
        Today = date.today().strftime("%Y-%m-%d")

        data_load_state = st.text('Loading data...')
        try:
            data = yf.download(selected_stock, Start, Today)
            t.sleep(2)
            data_load_state.text('Loading data... Done!')
        except:
            st.write('Enter a valid ticker')
            data_load_state.text('Loading data... Error!')

        df = data
        st.subheader('Raw Data Description')
        st.write(data.tail())

        # Preprocessing the data
        # Resample to daily frequency, forward fill missing values, and drop any remaining NaNs
        data = data['Close'].resample('D').ffill().dropna()

        # Scaling the data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(np.array(data).reshape(-1, 1))

        # Creating the features and target variables
        lookback = 60  # Number of days to look back
        X = []
        y = []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)

        # Splitting the data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # Reshaping the data for CNN
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        st.subheader("Showing 100 days and 200 days Moving Average")
        df = df.reset_index()
        # Moving average 100 calculation
        ma100 = df.Close.rolling(100).mean()
        # Moving average 200 calculation
        ma200 = df.Close.rolling(200).mean()

        # Chart for ma100 and ma200
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df.Close)
        plt.plot(ma100, 'r')
        plt.plot(ma200, 'g')
        st.pyplot(fig1)
        st.write(df.shape)
        st.write('If 100 days MA crosses 200 days MA from above, it means bearish reversal may take place and if 100 days MA crosses 200 days MA from below, it means bullish reversal may take place.')

        # load Model
        model = load_model('keras_CNN_model.h5')

        # Predicting the stock prices for the test set
        predicted_stock_prices = model.predict(X_test)
        predicted_stock_prices = scaler.inverse_transform(
            predicted_stock_prices)

        if st.button('Visualize Predictions'):
            # Visualizing the results
            st.subheader('Visualizing the Predicted Stock Prices Using CNN')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(scaler.inverse_transform(
                y_test.reshape(-1, 1)), label='Actual Stock Price')
            plt.plot(predicted_stock_prices, label='Predicted Stock Price')
            plt.legend()
            st.pyplot(fig2)
        else:
            st.write('Click on Visualize Predictions to get the further analysis.')

    if analysis == 'SVM Analysis':

        st.subheader("Using SVM Model")

        stock = st.text_input('Enter Stock Ticker For Prediction', 'META')
        st.write('Performing Analysis of ', stock)
        selected_stock = stock
        Start = "2015-01-01"
        Today = date.today().strftime("%Y-%m-%d")

        data_load_state = st.text('Loading data...')
        try:
            data = yf.download(selected_stock, Start, Today)
            t.sleep(2)
            data_load_state.text('Loading data... Done!')
        except:
            st.write('Enter a valid ticker')
            data_load_state.text('Loading data... Error!')

        df = data
        st.subheader('Raw Data Description')
        st.write(data.tail())

        # Preprocessing the data
        # Resample to daily frequency, forward fill missing values, and drop any remaining NaNs
        data = data['Close'].resample('D').ffill().dropna()
        # Scaling the data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(np.array(data).reshape(-1, 1))

        # Creating the features and target variables
        lookback = 60  # Number of days to look back
        X = []
        y = []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)

        # Splitting the data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # Creating the multilayer SVM model with RBF kernel
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

        # Fitting the model to the data
        model.fit(X_train, y_train)

        st.subheader("Showing 100 days and 200 days Moving Average")
        df = df.reset_index()
        # Moving average 100 calculation
        ma100 = df.Close.rolling(100).mean()
        # Moving average 200 calculation
        ma200 = df.Close.rolling(200).mean()

        # Chart for ma100 and ma200
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df.Close)
        plt.plot(ma100, 'r')
        plt.plot(ma200, 'g')
        st.pyplot(fig1)
        st.write(df.shape)
        st.write('If 100 days MA crosses 200 days MA from above, it means bearish reversal may take place and if 100 days MA crosses 200 days MA from below, it means bullish reversal may take place.')

        # Predicting the stock prices for the test set
        predicted_stock_prices = model.predict(X_test)
        predicted_stock_prices = scaler.inverse_transform(
            predicted_stock_prices.reshape(-1, 1))

        # Evaluating the model using mean squared error
        mse = mean_squared_error(y_test, predicted_stock_prices)
        rmse = np.sqrt(mse)
        st.write("Root Mean Squared Error:", rmse)

        if st.button('Visualize Predictions'):
            # Visualizing the results
            st.subheader('Visualizing the Predicted Stock Prices Using SVM')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(scaler.inverse_transform(
                y_test.reshape(-1, 1)), label='Actual Stock Price')
            plt.plot(predicted_stock_prices, label='Predicted Stock Price')
            plt.legend()
            st.pyplot(fig2)
        else:
            st.write('Click on Visualize Predictions to get the further analysis.')

if selected == "Know More":
    st.header("How to Predict Stock Price?")

    image2 = Image.open('images/analysis.jpg')
    st.image(image2)

    st.subheader("Methods of Fundamental Analysis and Technical Analysis")

    st.markdown("""Fundamental analysis is essentially getting to know the value of the stock by studying factors that could affect its price. 
        It could be internal or external factors. While internal factors could be anything like the financial health of the company, 
        future prospects, the market that it operates in, the management, the prospects of the sector it operates in and the overall 
        global and national economic conditions. <br> Technical analysis is the measure of the company's performance through certain 
        technical parameters. Technical analysis is the analysis of the current day's performance of a stock and based on 
        certain parameters to predict the movement of the stock in the following day. 
        This type of analysis is mostly used by expert analysts and not by average investors.""", True)

    st.subheader("Commonly used metrics in fundamental analysis are:")

    st.markdown("""**1. Earnings per share:** <br> 
        Earnings per share is the earnings that are earned by the shareholders for each share held by them. 
        A higher EPS than the industry average indicates that the company is performing better than its peers in the industry. 
        A company consistently providing higher EPS will be a preferred stock by the investors.""", True)

    st.markdown("""**2. Price to Earnings ratio** <br>
        Price to Earnings ratio is one of the traditional methods to analyse the company performance and predict the prices of the stock of the company. 
        This ratio considers the market price of the shares of the company and the earnings per share (EPS) of the company. 
        If the PE ratio is favourable than the industry standards, the company is considered to be in a better position than its peers. 
        It is a relatively outdated tool that is not used anymore by most analysts as a primary measure to predict the stock prices.""",
                True)

    st.markdown("""**3. Return on equity:** <br>
        The return on equity is one of the most important measures of a company's profitability. 
        A higher ROI will assure the investors of the profitability of the company and will eventually lead to an increase in the trade 
        volume and prices of the stock.""", True)

    st.markdown("""**4. Price to Earnings to growth ratio:** <br>
        Price to earnings to growth ratio is the addition to the price to earnings ratio. 
        This ratio provides a better yardstick to measure the performance of the company and thereby predict the prices of the stock. 
        The main feature of this ratio is that it considers the growth of the company to measure its performance and eventually predict 
        the stock prices the following day.""", True)

    st.subheader("Common metrics used in technical analysis are:")

    st.markdown("""**1. Simple Moving averages:** <br>
        By using this metric, you try to even out the day-to-day movements of the stock by taking averages for a certain number of days, 
        say 1 week, 10 days, 1month, 3 months etc. """, True)

    st.markdown("""**2. Exponential moving averages:** <br>
        An exponential moving average is a weighted moving average that assigns more importance to recent price movements than the older ones. """,
                True)

    st.markdown("""**3. Candlestick patterns:** <br>
        In this metric, candle stick like images are plotted for each day of trade for a stock. It involves data points like opening price, 
        closing price, the range, etc. When candlestick images are plotted for a number of days, there are patterns that emerge based 
        on which trading/investing decisions are taken.""", True)

    st.markdown("""**4. Momentum indicators:** <br>
        Several metrics that indicate the momentum of the stock like Stochastic Oscillator, Relative Strength Index, Moving Average Convergence 
        Divergence are also used as metrics to predict if the movement in stock prices is a change in trend or a range bound movement, 
        or an insignificant movement. """, True)

    st.markdown("""**5. Volume breakouts:** <br>
        This metric involves identifying a pattern when the stock breaks out of its set patterns with huge volumes. 
        This signifies a change in the trend of the stock price.""", True)

    st.subheader("Key Points To Remember")

    st.markdown("""**Risk Involved** <br> 
        There is no correct way to predict the stock prices with 100 percent accuracy. 
        Most expert analysts on many occasions fail to predict the stock prices or the prediction of movement of stock with even 60 percent to 80 percent accuracy. 
        Investors should consider multiple parameters to ensure that they can predict the stock price to the closest possible range and accordingly make investment decisions. \
        In most cases, the human intelligence factor is one of the most important decision making parameters in predicting the stock prices""", True)
