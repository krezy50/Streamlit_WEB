import streamlit as st
import pandas as pd
import datetime

import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from pandas_datareader import data as pdr
from scipy import stats
import numpy as np
import yfinance as yf

yf.pdr_override()

def MACDStrategy():
    """
    macd Oscillator 의 기울기로 매수 매도 비중을 조절한다.
    :return:
    """
    st.subheader("MACD, MACD Oscillator, RSI 활용한 주식투자 by 502")

    s1 = st.text_input("분석종목: ", value='SOXL')
    date = st.date_input("시작날짜", datetime.date(2022, 10, 1))

    # 실제 코드
    f1 = pdr.get_data_yahoo(s1, date)

    f1.index = f1.index.date  # index 시간제거
    df = pd.DataFrame(f1)

    ema12 = df.Close.ewm(span=12).mean()  # 종가의 12일 지수 이동평균
    ema26 = df.Close.ewm(span=26).mean()  # 종가의 26일 지수 이동평균
    macd = ema12 - ema26  # macd 선
    signal = macd.ewm(span=9).mean()  # 신호선 (macd의 9일 지수 이동평균)
    macdhist = macd - signal

    period=14
    delta = df.Close.diff()
    up, down = delta.copy(), delta.copy()
    up[up<0]=0
    down[down>0]=0
    gain = up.ewm(com=(period-1), min_periods=period).mean()
    loss = down.abs().ewm(com=(period-1), min_periods=period).mean()
    RS = gain/loss
    rsi= 100 - (100/(1+RS))

    df = df.assign(ema26=ema26, ema12=ema12, macd=macd, signal=signal, macdhist=macdhist, rsi=rsi).dropna()

    df['number'] = df.index.map(mdates.date2num)  # 캔들차트에 사용할수 있게 날짜형 인덱스를 숫자형으로 변환
    ohlc = df[['number', 'Open', 'High', 'Low', 'Close']]

    plt.figure(figsize=(9, 9))

    p1 = plt.subplot(3, 1, 1)
    plt.title(f'MACD / MACD Osillator / RSI Trading {s1}')
    plt.grid(True)
    candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red',
                     colordown='blue')  # ohlc의 숫자형 일자,시가,고가,저가,종가 값을 이용해서 캔들차트를 그린다.
    p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot(df.number, df['ema26'], color='c', label='EMA26')
    plt.plot(df.number, df['ema12'], color='g', label='EMA12',linestyle ='--')
    plt.legend(loc='best')

    p2 = plt.subplot(3, 1, 2)
    plt.grid(True)
    p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.bar(df.number, df['macdhist'], color='m', label='MACD-Hist')
    plt.plot(df.number, df['macd'], color='b', label='MACD')
    plt.plot(df.number, df['signal'], 'g--', label='MACD-signal')\

    for i in range(1, len(df.Close)):
        if df.macdhist.values[i - 1] < df.macdhist.values[i] and df.rsi.values[i]<35 :
            plt.plot(df.number.values[i], 0, 'r^')
        elif df.macdhist.values[i - 1] > df.macdhist.values[i] and df.rsi.values[i]>65 :
            plt.plot(df.number.values[i], 0, 'bv')

    plt.legend(loc='best')

    p2 = plt.subplot(3, 1, 3)
    plt.grid(True)
    p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot(df.number, df['rsi'], color='b', label='%RSI')
    plt.axhline(65,0,1,color = 'green', linestyle ='--')
    plt.axhline(35,0,1,color = 'green', linestyle ='--')
    # plt.yticks([0, 20, 80, 100])
    plt.legend(loc='best')
    figure = plt.show()
    st.pyplot(figure)
