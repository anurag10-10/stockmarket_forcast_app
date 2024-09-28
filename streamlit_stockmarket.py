import streamlit as st
from datetime import date
import requests
import pandas as pd
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly

# Sample data for stock symbols, ISINs, and other information
data = {
    'Symbol': ['BAJAJ-AUTO', 'BLACKROSE-X', 'BOSCHLTD', 'DEEPAKNTR', 'HAL', 'HAPPSTMNDS', 'HDFCBANK', 'IDFCFIRSTB', 'INFY', 'INTELLECT', 'IRB', 'LTIM', 'LXCHEM', 'ONGC', 'PERSISTENT', 'PRINCEPIPE', 'RVNL', 'SJVN', 'SUZLON', 'TATACONSUM', 'TATAPOWER', 'WIPRO'],
    'ISIN': ['INE917I01010', 'INE761G01016', 'INE323A01026', 'INE288B01029', 'INE066F01020', 'INE419U01012', 'INE040A01034', 'INE092T01019', 'INE009A01021', 'INE306R01017', 'INE821I01022', 'INE214T01019', 'INE576O01020', 'INE213A01029', 'INE262H01021', 'INE689W01016', 'INE415G01027', 'INE002L01015', 'INE040H01021', 'INE192A01025', 'INE245A01021', 'INE075A01022']
}

# Create a DataFrame from the data
stock_df = pd.DataFrame(data)

# Streamlit app starts here
st.title("Stock Price Data")

# Select box for choosing stock from the list
stock_name = st.selectbox("Select a Stock Symbol", options= list(stock_df["Symbol"]) + ["Other"])

# Initialize ISIN to None
selected_isin = None

# If "Other" is selected, show an input box for the user to type the ISIN
if stock_name == "Other":
    stock_name = st.text_input("Enter the stock name:")
    selected_isin = st.text_input("Enter the ISIN code")
elif stock_name != "Select a stock...":
    # Find the corresponding ISIN for the selected stock symbol
    selected_isin = stock_df.loc[stock_df["Symbol"] == stock_name, "ISIN"].values[0]

# If no valid ISIN is provided, stop the execution
if not selected_isin:
    st.warning("Please select a stock or enter a valid ISIN.")
    st.stop()

# Continue only if a valid ISIN is provided
st.write(f"Selected stock is {stock_name} with ISIN: {selected_isin}")

# Selectbox to choose the timeframe
timeframe = ['1minute', 'day', 'week', 'month']
timeframe = st.selectbox('Select a data timeframe:', timeframe)

# Slider for the number of years for prediction
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Define the URL for the API request using the selected ISIN
url = f'https://api.upstox.com/v2/historical-candle/NSE_EQ%7C{selected_isin}/{timeframe}/2024-09-19/2000-11-12'

# Make the API request
response = requests.get(url, headers={'Accept': 'application/json'})

if response.status_code == 200:
    # Parse the response JSON
    data = response.json()

    # Check if 'candles' key is in the response
    if 'candles' in data['data']:
        # Extract the 'candles' data
        candles = data['data']['candles']

        # Create a DataFrame from the 'candles' data
        df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unknown'])

        df.insert(df.columns.get_loc('Timestamp'), 'Date', pd.to_datetime(df['Timestamp']).dt.date)
        df = df.drop(columns=['Timestamp','Unknown'])


        # Display the DataFrame
        st.write(df)
    else:
        st.write("No candle data available")
else:
    st.write(f"Error: {response.status_code} - {response.text}")

# Ensure df exists before proceeding
if 'df' in locals():
    df['DS'] = pd.to_datetime(df['Date']).dt.date

    @st.cache_data
    def plot_raw_data(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['DS'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['DS'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data(df)

    df.rename(columns={'Close': 'Y'}, inplace=True)
    df_formatted = df[['DS', 'Y']]

    # Predict forecast with Prophet
    df_train = df_formatted.rename(columns={"DS": "ds", "Y": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast, xlabel='Date', ylabel='Price')
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig3 = m.plot_components(forecast)
    st.write(fig3)
