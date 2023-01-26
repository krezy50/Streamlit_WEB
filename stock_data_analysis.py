import streamlit as st
import pandas as pd
import datetime


import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from scipy import stats
import numpy as np
import yfinance as yf
yf.pdr_override()


def CompareStockAnalysis():
    # st.markdown("일간 변동률로 주가 비교하기")
    st.write("일간 변동률로 주가 비교하기",
             "오늘변동률 = ((오늘종가- 어제종가)/어제종가)*100 : 주가가 상이한 종목별을 비교할때 이용"
             "일간 변동률 누적곱 구하여 전체적인 변동률을 비교할수 있다. cumprod()함수 활용 ")

    stock1 = st.text_input("비교 종목 : ", value='AAPL')
    stock2 = st.text_input("비교 종목 : ", value='MSFT')
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

def RelationAnalysis():

    st.write("회귀분석은 데이터의 상관관계를 분석하는데 쓰이는 통계분석방법이다. "
             "회귀분석은 회귀목형을 설정한 후 실제로 관측된 표본을 대상으로 회귀 모형의 계수를 추정한다. "
             "독립변수라고부리는 하나 이상의 변수와 종속변수라 불리는 하나의 변수 간의 관계를 나타내는 회귀식이 도출되면, "
             "임의의 독립변수에 대하여 종속변숫값을 추측해 볼수 있는데, 이를 예측이라고한다.")
    stock1 = st.text_input("비교 종목 1: ", value='AAPL')
    stock2 = st.text_input("비교 종목 2: ", value='ABBV')

    date = st.date_input("시작날짜 입력", datetime.date(2013, 1, 2))

    fdata1 = pdr.get_data_yahoo(stock1,date)
    fdata2 = pdr.get_data_yahoo(stock2,date)

    fdata1.index = fdata1.index.date #index 시간제거
    fdata2.index = fdata2.index.date #index 시간제거

    d = (fdata1.Close / fdata1.Close.iloc[0]) * 100 # 지수화
    #시작날짜기준으로 지수를 나눠 100을 곱한다. 두 주식간의 상승률을 비교할수 있다.
    k = (fdata2.Close / fdata2.Close.iloc[0]) * 100 # 지수화

    plt.figure(figsize=(9,5))
    plt.plot(d.index,d,'r--',label=stock1)
    plt.plot(k.index,k,'b',label=stock2)
    plt.grid(True)
    plt.legend(loc='best')
    figure = plt.show()
    st.pyplot(figure)

    st.write("산점도란 독립변수x와 종속변수y의 상관관계를 확인할때 쓰는 그래프다. 가로축은 독립변수x를, 세로축은 종속변수를 나타낸다."
             "x,y 개수는 동일해야함. y=x인 직선형태에 가까울수로 직접적인 관계가 있다.")

    # st.write(fdata1['Close'],fdata2['Close'])
    # st.write(len(fdata1),len(fdata2))
    df = pd.DataFrame({f'{stock1}':fdata1['Close'],f'{stock2}':fdata2['Close']}) #합치기
    df = df.fillna(method='bfill') #bfill(backward fill), ffill(forward fill)
    df = df.fillna(method='ffill')
    #dropna()은 nan 있는 행을 삭제

    # plt.figure(figsize=(7,7))
    # plt.scatter(df[f'{stock1}'],df[f'{stock2}'],marker='.')
    # plt.xlabel(f'{stock1}')
    # plt.ylabel(f'{stock2}')
    # figure = plt.show()
    # st.pyplot(figure)


    #LinearRegressionModel
    st.write("회귀모델이란 연속적인 Y와 이 Y의 원인이 되는 X간의 관계를 추정하는 관계식을 의미"
             "실제로 데이터 값에는 측정상의 한계로 인한 잡음이 존재하기 때문에 정확한 관계식을 표현하는 확률변수인 오차항을 두게된다.")

    regr = stats.linregress(df[f'{stock1}'],df[f'{stock2}'])
    st.write(regr)
    st.write("slope : 기울기, intercept : 절편, rvalue : r값(상관계수), pvalue : p값, stderr : 표준편차"
             "stats로 생성한 모델을 이용하면 선형회귀식을 구할수 있다.")

    r_value = df[f'{stock1}'].corr(df[f'{stock2}'])
    r_squared = r_value**2
    st.write("상관계수 : ",r_value)
    st.write("결정계수는 관측된 데이터에서 추정한 회귀선이 실제로 데이터를 어느정도 설명하는지를 나타내는 계수로 "
             "두변수의 상관관계정도를 나타내는 상관계수를 제곱한 값이다. "
             "결정계수가 1이면 모든 표본관측치가 추정된 회귀선 상에만 있다는 의미다. 즉 추정된 회귀선이 변수간의 관계를 완벽히설명"
             "반면에 0이면 추정된 회귀선이 변수사이의 관계를 전혀 설명하지 못한다는 의미다.")
    st.write("결정계수 : ", r_squared)

    regr_line = f'Y = {regr.slope:.2f} * X + {regr.intercept:.2f}'
    plt.figure(figsize=(7,7))
    plt.plot(df[f'{stock1}'],df[f'{stock2}'],'.')
    plt.plot(df[f'{stock1}'],regr.slope * df[f'{stock1}']+regr.intercept,'r')
    plt.legend([f'{stock1}x{stock2}',regr_line])
    plt.title(f'{stock1}x{stock2} (R={regr.rvalue:.2f})')
    plt.xlabel(f'{stock1}')
    plt.ylabel(f'{stock2}')
    figure = plt.show()
    st.pyplot(figure)


    st.write("트레이딩 전략 구현")
    #시총상위 종목으로 효율적 투자선 구하기
    s1 = st.text_input("비교 종목1: ", value='AAPL')
    s2 = st.text_input("비교 종목2: ", value='ABBV')
    s3 = st.text_input("비교 종목3: ", value='SOXL')
    s4 = st.text_input("비교 종목4: ", value='NAIL')

    date = st.date_input("시작날짜", datetime.date(2013, 1, 2))

    f1 = pdr.get_data_yahoo(s1,date)
    f2 = pdr.get_data_yahoo(s2,date)
    f3 = pdr.get_data_yahoo(s3,date)
    f4 = pdr.get_data_yahoo(s4,date)

    f1.index = f1.index.date #index 시간제거
    f2.index = f2.index.date #index 시간제거
    f3.index = f3.index.date #index 시간제거
    f4.index = f4.index.date #index 시간제거

    df = pd.DataFrame({f'{s1}': f1['Close'], f'{s2}': f2['Close'], f'{s3}': f3['Close'], f'{s4}': f4['Close']})  # 합치기
    df = df.fillna(method='bfill')  # bfill(backward fill), ffill(forward fill)
    df = df.fillna(method='ffill')

    daily_ret = df.pct_change() #일간 변동률
    annual_ret = daily_ret.mean()*252 #연간수익률
    daily_cov = daily_ret.cov() # 일간 변동률의 공분산
    annual_cov = daily_cov*252 #연간 공분산

    port_ret =[]
    port_risk =[]
    port_weights=[]

    st.write("몬테카를로 시뮬레이션 : 매우 많은 난수를 이용해 함수의 값을 확률적으로 계산하는 것, "
             "4개 종목의 비중을 난수를 이용해서 20000개의 포트폴리오 생성하여 수익률과 리스크를 분석 ")

    stock = [f'{s1}',f'{s2}',f'{s3}',f'{s4}']

    for _ in range(20000): #반복횟수를 사용할일이 없을 경우 _
        weights = np.random.random(len(stock))  # 4개의 랜덤숫자로 구성된 배열을 생성
        weights /= np.sum(weights) #4개의 랜덤숫자를 랜덤숫자 총합으로 나눠 4종목의 비중의합이 1이 되도록 조정

        # 랜덤하게 생성한종목별 비중 배열과 종목별 연간수익률을 곱해 해당 포트폴리오 전체 수익률을 구한다.
        returns = np.dot(weights,annual_ret)

        # 종목별 연간 공분산과 종목별 비중 배열을 곱한뒤 이를 다시 종목별 비중의 전치로 곱한다.
        # 이렇게 구한 결과값의 제곱근 sqrt() 구하면 해당 포트폴리오의 전체 리스크를 구할수 있다.
        risk = np.sqrt(np.dot(weights.T,np.dot(annual_cov,weights)))

        port_ret.append(returns)
        port_risk.append(risk)
        port_weights.append(weights)

    portfolio = {'Returns':port_ret, 'Risk':port_risk}

    for i,s in enumerate(stock): #각 종목별로 비중 입력
        portfolio[s]=[weights[i] for weights in port_weights]

    df = pd.DataFrame(portfolio)
    df = df[['Returns','Risk'] + [s for s in stock]]

    df.plot.scatter(x='Risk',y='Returns',figsize=(10,7),grid=True)
    plt.title('Efficient Frontier')
    plt.xlabel('Risk')
    plt.ylabel('Expected Returns')
    figure = plt.show()
    st.pyplot(figure)