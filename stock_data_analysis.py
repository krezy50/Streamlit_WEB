import streamlit as st
import pandas as pd


import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


def CompareStockAnalysis():
    # st.markdown("일간 변동률로 주가 비교하기")
    st.write("일간 변동률로 주가 비교하기",
             "오늘변동률 = ((오늘종가- 어제종가)/어제종가)*100 : 주가가 상이한 종목별을 비교할때 이용"
             "일간 변동률 누적곱 구하여 전체적인 변동률을 비교할수 있다. cumprod()함수 활용 ")

    stock1 = st.text_input("비교 종목 1: ", value='AAPL')
    stock2 = st.text_input("비교 종목 2: ", value='MSFT')
    date = st.text_input("시작날짜 입력", value='2018-05-04')


    first = pdr.get_data_yahoo(stock1, start=date)
    first_dpc = (first['Close'] - first['Close'].shift(1)) / first['Close'].shift(1) * 100
    first_dpc.iloc[0] = 0
    first_dpc_cp = ((100 + first_dpc) / 100).cumprod() * 100 - 100  # 일간 변동률 누적곱 계산

    second = pdr.get_data_yahoo(stock2, start=date)
    second_dpc = (second['Close'] - second['Close'].shift(1)) / second['Close'].shift(1) * 100
    second_dpc.iloc[0] = 0
    second_dpc_cp = ((100 + second_dpc) / 100).cumprod() * 100 - 100  # 일간 변동률 누적곱 계산

    plt.plot(first.index, first_dpc_cp, 'b', label=stock1)
    plt.plot(second.index, second_dpc_cp, 'r--', label=stock2)
    plt.ylabel('Change %')
    plt.grid(True)
    plt.legend(loc='best')
    figure = plt.show()
    st.pyplot(figure)


def MDDAnalysis():
    
    st.write("MDD 최대손실낙폭은 특정기간에 발생한 최고점에서 최점까지의 가장 큰손실을 의미")
    st.write("MDD = (최저점 - 최고점)/최저점")
    #rolling 함수는 시리즈에서 윈도우 크기에 해당하는 개수만큼 데이터를 추출하여 집계함수에 해당하는 연산을 실시
    #집계함수로는 최대값,평균값,최소값을사용

    stock = st.text_input("종목", value='^KS11')
    date = st.text_input("시작날짜 입력", value='2004-01-04')

    temp = pdr.get_data_yahoo(stock,date)

    window = 252 #1년 단위의 window설정
    peak = temp['Adj Close'].rolling(window,min_periods=1).max()
    drawdown = temp['Adj Close']/peak - 1.0 #최고치(peak) 대비 현재 KOSPI종가가 얼마나 하락했는지를 구한다.
    max_dd = drawdown.rolling(window,min_periods=1).min() #1년 기간단위로 최저치 MDD를 구한다.

    plt.figure(figsize=(9,7))
    plt.subplot(211)
    temp['Close'].plot(label=f'{stock}', title='KOSPI MDD',grid=True,legend=True)
    plt.subplot(212)
    drawdown.plot(c='blue',label=f'{stock} DD',grid=True,legend=True)
    max_dd.plot(c='red',label=f'{stock} MDD',grid=True,legend=True)
    figure = plt.show()
    st.pyplot(figure)
    st.write('max_dd(MDD):',format(max_dd.min()))
    st.write('MDD 기간',max_dd[max_dd==max_dd.min()])