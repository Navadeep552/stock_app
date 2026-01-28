import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# ---------------- SETTINGS ----------------
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("📈 STOCK PRICE PREDICTION APP")

# ---------------- STOCK SELECTION ----------------
stocks = ("AAPL", "GOOG", "MSFT", "GME", "TSLA", "AMZN", "INFY.NS", "TCS.NS")
selected_stock = st.selectbox("Select Stock Symbol", stocks)

n_years = st.slider("Years of Prediction:", 1, 6)
period = n_years * 365

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if data.empty:
        return None

    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

data = load_data(selected_stock)

# ---------------- MAIN APP ----------------
if data is not None:

    data['Date'] = pd.to_datetime(data['Date'])

    # -------- RAW DATA --------
    st.subheader("📄 Raw Stock Data (Last 5 Rows)")
    st.write(data.tail())

    # -------- DATE SEARCH --------
    st.subheader("🔍 Search Stock Data by Date")

    search_date = st.date_input(
        "Select a date",
        min_value=pd.to_datetime(START),
        max_value=pd.to_datetime(TODAY)
    )

    selected_day_data = data[data['Date'] == pd.to_datetime(search_date)]

    if not selected_day_data.empty:
        row = selected_day_data.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Open", round(row['Open'], 2))
        col2.metric("Close", round(row['Close'], 2))
        col3.metric("High", round(row['High'], 2))
        col4.metric("Low", round(row['Low'], 2))

        st.write(selected_day_data)

    else:
        st.warning("⚠️ No trading data for this date (holiday or weekend).")

    # -------- PRICE GRAPH --------
    st.subheader("📊 Stock Price Chart")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
    fig.update_layout(title="Stock Price Over Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    # -------- PROPHET PREDICTION --------
    st.subheader("🔮 Stock Price Forecast")

    df_train = data[['Date', 'Close']].copy()
    df_train.dropna(inplace=True)
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader("📈 Forecast Data (Last 5 Rows)")
    st.write(forecast.tail())

    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📉 Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

else:
    st.error("❌ Failed to load stock data. Please try another symbol.")