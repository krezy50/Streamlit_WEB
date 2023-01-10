# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import pandas as pd

from rental_investment_calculator import RentalInvestmentCalculator #임대 수익 계산기
from market_rate import MarketRateScrapping #스크랩핑
from control_excel import convert_df

from backtesting import Backtest
from stock_back_test import SmaCross,RSIStrategy,BBStrategy,DonchainStrategy,MACDCross
import FinanceDataReader as fdr
import datetime

#TA-Lib install StreamLit
import requests
import os
import sys
import subprocess

# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    os.system("ls -la /app/equity/")
    # build
    os.system("./configure --prefix=/home/appuser")
    os.system("make")
    # install
    os.system("make install")
    # back to the cwd
    os.chdir(default_cwd)
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
# import library
try:
    import talib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/", "--global-option=-I/home/appuser/include/", "ta-lib"])
finally:
    import talib

#TA-Lib install StreamLit



with st.form("시스템 선택"):
    st.header("Python projects of 502")

    system=st.radio("Choice a project", ('임대 수익률 계산기', '시장 금리 스크래핑','BackTesing'))
    submitted = st.form_submit_button("Submit")

if system == '임대 수익률 계산기':
    st.caption("Sidebar MENU 에서 관련 정보를 입력하세요.")
    result = RentalInvestmentCalculator()

    title = st.text_input("다운받을 파일명을 입력하세요.",max_chars=10)
    add_title = result.rename(columns={0:title}) #제목 index으로 변경
    csv = convert_df(add_title)

    st.download_button( #파일에 쓰고 다운로드
        label="Download data as CSV",
        data=csv,
        file_name=title+".csv",
        mime='text/csv',
    )

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

elif system == '시장 금리 스크래핑':

    MarketRateScrapping()

elif system == 'BackTesing':

    st.markdown("https: // kernc.github.io / backtesting.py /")

    selected_stock_value = st.text_input('Input STOCK CODE to use in backtest', value='SOXL')

    start_date = st.date_input("Choice a Start Day",datetime.date(2015, 1, 1))
    price_df = fdr.DataReader(selected_stock_value, start_date)

    # Set Strategy Parameters

    strategy_dict = {
        "Moving Average Crossover": SmaCross,
        "Relative Strength Index": RSIStrategy,
        "Bollinger Band": BBStrategy,
        "Donchain Channel": DonchainStrategy,
        "MACD Cross":MACDCross,
    }

    # Select a Strategy
    selected_strategy_key = st.selectbox('Select a strategy', list(strategy_dict.keys()))
    selected_strategy = strategy_dict[selected_strategy_key]

    params = dict()

    if selected_strategy_key == "Moving Average Crossover":
        short_term = st.number_input("Set Short-term Moving Average Lookback Period", value=10)
        long_term = st.number_input("Set Long-term Moving Average Lookback Period", value=20)
        params['short_term'] = short_term
        params['long_term'] = long_term

    elif selected_strategy_key == "Relative Strength Index":
        lookback_period = st.number_input("Set RSI Lookback Period", value=14)
        buy_level = st.number_input("Set RSI Buy Level", value=50)
        sell_level = st.number_input("Set RSI Sell Level", value=50)
        params['lookback_period'] = lookback_period
        params['buy_level'] = buy_level
        params['sell_level'] = sell_level

    elif selected_strategy_key == "Bollinger Band":
        lookback_period = st.number_input("Set Bollinger Band Lookback Period", value=20)
        params['lookback_period'] = lookback_period

    elif selected_strategy_key == "Donchain Channel":
        lookback_period = st.number_input("Set Donchain Channel Lookback Period", value=100)
        params['lookback_period'] = lookback_period

    elif selected_strategy_key == "MACD Cross":
        short_term = st.number_input("Set Short-term MACD Lookback Period", value=12)
        long_term = st.number_input("Set Long-term MACD Lookback Period", value=26)
        sequence = st.number_input("Set sequence MACD Lookback Period", value=9)
        params['short_term'] = short_term
        params['long_term'] = long_term
        params['sequence'] = sequence

    cost = st.number_input("Set Transaction Cost (%)", value=0.1) * 0.01

    st.write(price_df)
    #Backtest(주가정보, 전략, 진입 주식수, 거래세, 등)
    bt = Backtest(price_df, selected_strategy,
                  cash=1000000, commission=cost,
                  trade_on_close=True,
                  exclusive_orders=True)
    #통계수치보기
    output = bt.run(**params)
    # output = bt.run()


    #그래프 보기
    output_df = pd.DataFrame(output)
    st.dataframe(output_df[:-2], height=800)

    # 그래프 에러 개선
    # pip install bokeh==2.4.3
    # https://github.com/kernc/backtesting.py/issues/803
    # https://stackoverflow.com/questions/74334910/backtesting-py-ploting-function-not-working

    bt.plot(open_browser=False, filename="backtest_plot")
    with open("backtest_plot.html", "r", encoding='utf-8') as f:
        plot_html = f.read()
    st.components.v1.html(plot_html, height=1000)
