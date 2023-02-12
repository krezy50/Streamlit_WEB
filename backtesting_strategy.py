import streamlit as st
import pandas as pd

from backtesting.test import SMA
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest

import FinanceDataReader as fdr
import datetime

# import tulipy as ti (C 언어 기반으로 streamlit 에 설치불가)
# import talib as ta (C 언어 기반으로 streamlit 에 설치불가)
# TA-lib 설치 방법 (Visual Studio Community 를 설치 후에 파워셀에서 컴파일 후 pip install 해야함)
# https://github.com/minggnim/ta-lib
# Download and Unzip ta-lib-0.4.0-msvc.zip
# Move the Unzipped Folder ta-lib to C:\
# Download and Install Visual Studio Community (2015 or later)
# Remember to Select [Visual C++] Feature
# Build TA-Lib Library
# From Windows Start Menu, Start [VS2015 x64 Native Tools Command Prompt]
# Move to C:\ta-lib\c\make\cdr\win32\msvc
# Build the Library nmake

# TA-LIB 대체 라이브러리
# https://github.com/twopirllc/pandas-ta 51
# https://github.com/bukosabino/ta 22
# https://github.com/peerchemist/finta 15


# 전략 구현 - 1. Moving Average
# 이동평균선이란 N일 동안의 주가를 평균한 값으로 이어진 선을 의미한다.
# 단기 이동평균선이 장기 이동평균선을 상승돌파하면 매수, 하강돌파하면 매도
# 기본값은 단기 이동평균선 short_term 10일, 장기 이동평균선 long_term 20일

class SmaCross(Strategy): #전략이름

    #파라미터 value 설정
    short_term = 10
    long_term = 20

    def init(self):
        #지표는 TA-Lib 이나 Tulipy 에서 불러보자

        close = self.data.Close
        #self.지표 = self.I(지료, 종가(혹은 시가, 고가,저가 중 필요한 것, 파라미터 value)
        self.sma1 = self.I(SMA, close, self.short_term)
        self.sma2 = self.I(SMA, close, self.long_term)

    def next(self):
        #if 내가 원하는 진입 조건:
        if crossover(self.sma1, self.sma2):
            self.buy()
        #elif 탈출 조건:
        elif crossover(self.sma2, self.sma1):
            self.sell()


# 전략 구현 - 2. Relative Strength Indicator
# RSI는 매수/매도의 강도를 나타내는 지표.
# 70% 이상을 초과매수 국면으로, 30% 이하를 초과매도 국면으로 보고 각각 매도, 매수 시그널로 만들어 역추세 매매기법을 하기도 한다.
# 여기에서는 추세 매매를 위해 50%를 상승돌파하면 매수, 50%를 하향돌파하면 매도하는 전략을 사용한다.
# 기본값은 RSI 계산을 위한 lookback_period 14일, buy_level=50, sell_level=50

def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)

class RSIStrategy(Strategy):

    lookback_period = 14
    buy_level = 50
    sell_level = 50

    def init(self, lookback_period=14):
        # Compute moving averages the strategy demands
        self.ma10 = self.I(SMA, self.data.Close, 10)
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma50 = self.I(SMA, self.data.Close, 50)
        self.ma100 = self.I(SMA, self.data.Close, 100)

        # Compute daily RSI
        self.daily_rsi = self.I(RSI, self.data.Close, self.lookback_period)

    def next(self):
        price = self.data.Close[-1]

        # If we don't already have a position, and
        # if all conditions are satisfied, enter long.
        if self.daily_rsi[-1] > self.buy_level and not self.position.is_long:
            self.buy()

        elif self.daily_rsi[-1] < self.sell_level and not self.position.is_short:
            self.sell()

# 전략 구현 - 3. Bollinger Band
# 볼린저밴드란 이동평균선을 중심으로 주가 변동의 표준편차를 더하여 상단밴드, 빼서 하단밴드를 만들어낸 것을 말한다.
# 여기서는 표준편차 1을 설정하여 상단밴드를 상승돌파하면 매수, 하단밴드를 하향돌파하면 매도하는 전략을 사용한다.
# 기본값은 볼린저밴드 계산을 위한 lookback_period 20일

def BB(array, n, is_upper):
    sma = pd.Series(array).rolling(n).mean()
    std = pd.Series(array).rolling(n).std()
    upper_bb = sma + std * 1
    lower_bb = sma - std * 1
    if is_upper:
        return upper_bb
    else:
        return lower_bb

class BBStrategy(Strategy):

    lookback_period = 20

    def init(self):
        # Compute daily Bollinger Band
        self.upper_bb = self.I(BB, self.data.Close, self.lookback_period, True)
        self.lower_bb = self.I(BB, self.data.Close, self.lookback_period, False)

    def next(self):
        price = self.data.Close[-1]

        if self.upper_bb[-1] < price and not self.position.is_long:
            self.buy()

        elif self.lower_bb[-1] > price and not self.position.is_short:
            self.sell()

# 전략 구현 - 4. Donchain Channel
# 상단밴드는 지난 n일 간의 가격 중 가장 높았던 가격, 하단밴드는 지난 n일 간의 가격 중 가장 낮았던 가격을 말한다. 이 밴드들은 보통 뚫기 어려운 저항선이 된다.
# 여기서는 상단밴드 이상으로 가격이 올라가면 매수, 하단밴드 이하로 가격이 내려가면 매도하는 전략을 사용한다.
# 기본값은 Donchain channel 계산을 위한 lookback_period 20일

def Donchain(array, n, is_upper):
    rolling_max = pd.Series(array).rolling(n).max()
    rolling_min = pd.Series(array).rolling(n).min()
    if is_upper:
        return rolling_max
    else:
        return rolling_min

class DonchainStrategy(Strategy):

    lookback_period = 100

    def init(self):
        # Compute Donchain Channel
        self.upper_dc = self.I(Donchain, self.data.Close, self.lookback_period, True)
        self.lower_dc = self.I(Donchain, self.data.Close, self.lookback_period, False)

    def next(self):
        price = self.data.Close[-1]

        if self.upper_dc[-1] <= price and not self.position.is_long:
            self.buy()
        elif self.lower_dc[-1] >= price and not self.position.is_short:
            self.sell()

def MACD(close, n1, n2, ns):
    # n1-n2
    # TA-LIB 적용기준
    macd, macdsignal, macdhist = ta.MACD(close, fastperiod=n1, slowperiod=n2, signalperiod=ns)

    return macd, macdsignal

class MACDCross(Strategy):

    #파라미터 value 설정
    short_term = 12
    long_term = 26
    sequence = 9

    def init(self):
        close = self.data.Close
        self.macd, self.signalma = self.I(MACD, close, self.short_term, self.long_term, self.sequence)

    def next(self):
        if crossover(self.macd, self.signalma):
            self.buy()
        elif crossover(self.signalma, self.macd):
            self.position.close()


def Backtesting():
    selected_stock_value = st.text_input('Input STOCK CODE to use in backtest', value='SOXL')

    start_date = st.date_input("Choice a Start Day",datetime.date(2015, 1, 1))
    price_df = fdr.DataReader(selected_stock_value, start_date)

    # Set Strategy Parameters

    strategy_dict = {
        "Moving Average Crossover": SmaCross,
        "Relative Strength Index": RSIStrategy,
        "Bollinger Band": BBStrategy,
        "Donchain Channel": DonchainStrategy,
        # "MACD Cross":MACDCross,
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

    # elif selected_strategy_key == "MACD Cross":
    #     short_term = st.number_input("Set Short-term MACD Lookback Period", value=12)
    #     long_term = st.number_input("Set Long-term MACD Lookback Period", value=26)
    #     sequence = st.number_input("Set sequence MACD Lookback Period", value=9)
    #     params['short_term'] = short_term
    #     params['long_term'] = long_term
    #     params['sequence'] = sequence

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
