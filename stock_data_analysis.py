import streamlit as st
import pandas as pd
import datetime


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
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

def MonteCarloSimulation():

    st.subheader("Harry Max Markowitz - 1952년 포트폴리오 셀렉션 제시, 1990년 현대 포트폴리오 이론 창안")

    st.write("그가 제시한 평균 - 분산 최적화는 예상수익률과 리스크의 상관관계를 활용해 포트폴리오를 최적화하는것")
    st.write("몬테카를로 시뮬레이션 : 매우 많은 난수를 이용해 함수의 값을 확률적으로 계산하는 것, "
             "4개 종목의 비중을 난수를 이용해서 20000개의 포트폴리오 생성하여 수익률과 리스크를 분석 ")
    st.write("효율적 투자선기준(투자자가 인내할수있는 리스크수준에서 최상의 수익률을 제공하는 포트폴리오 집합)으로 자산을 배분")
    #시총상위 종목으로 효율적 투자선 구하기
    s1 = st.text_input("포트폴리오 종목1: ", value='AAPL')
    s2 = st.text_input("포트폴리오 종목2: ", value='ABBV')
    s3 = st.text_input("포트폴리오 종목3: ", value='SOXL')
    s4 = st.text_input("포트폴리오 종목4: ", value='NAIL')

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

def SharpRatioSimulation():
    st.subheader("William Sharpe - 마코위츠의 제자")

    st.write("Sharp Ratio 샤프지수는 측정된 위험 단위당 수익률을 계산")
    st.write(":green[$샤프지수 = \dfrac{포트폴리오 예상 수익률 - 무위험률}{수익률의 표준편차}$]")
    st.write("무위험률은 계산편의를 위해 0으로 가정, 샤프지수는 포트폴리오의 예상수익률을 수익률의 표준편차로 나누어서 구한다.")
    st.write("예를 들어 예상수익률이 7%이고 수익률의 표준편차가 5%인 경우 샤프지수는 7 $\div$ 5 = 1.4가 된다.")
    st.write("샤프지수가 높을수록 위험에 대한 보상이 더 크다.")

    s1 = st.text_input("포트폴리오 종목1: ", value='CURE')
    s2 = st.text_input("포트폴리오 종목2: ", value='SOXL')
    s3 = st.text_input("포트폴리오 종목3: ", value='NAIL')
    s4 = st.text_input("포트폴리오 종목4: ", value='TQQQ')
    s5 = st.text_input("포트폴리오 종목5: ", value='AAPL')

    date = st.date_input("시작날짜", datetime.date(2010, 1, 2))

    f1 = pdr.get_data_yahoo(s1,date)
    f2 = pdr.get_data_yahoo(s2,date)
    f3 = pdr.get_data_yahoo(s3,date)
    f4 = pdr.get_data_yahoo(s4,date)
    f5 = pdr.get_data_yahoo(s5,date)

    f1.index = f1.index.date #index 시간제거
    f2.index = f2.index.date #index 시간제거
    f3.index = f3.index.date #index 시간제거
    f4.index = f4.index.date #index 시간제거
    f5.index = f5.index.date #index 시간제거

    df = pd.DataFrame({f'{s1}': f1['Close'], f'{s2}': f2['Close'], f'{s3}': f3['Close'], f'{s4}': f4['Close'], f'{s5}': f5['Close']})  # 합치기
    df = df.fillna(method='bfill')  # bfill(backward fill), ffill(forward fill)
    df = df.fillna(method='ffill')

    daily_ret = df.pct_change() #일간 변동률
    annual_ret = daily_ret.mean()*252 #연간수익률
    daily_cov = daily_ret.cov() # 일간 변동률의 공분산
    annual_cov = daily_cov*252 #연간 공분산

    port_ret =[]
    port_risk =[]
    port_weights=[]
    sharp_ratio =[] #샤프 지수 추가

    stock = [f'{s1}',f'{s2}',f'{s3}',f'{s4}',f'{s5}']

    for _ in range(20000): #반복횟수를 사용할일이 없을 경우 _
        weights = np.random.random(len(stock))  # 랜덤숫자로 구성된 배열을 생성
        weights /= np.sum(weights) #4개의 랜덤숫자를 랜덤숫자 총합으로 나눠 4종목의 비중의합이 1이 되도록 조정

        # 랜덤하게 생성한종목별 비중 배열과 종목별 연간수익률을 곱해 해당 포트폴리오 전체 수익률을 구한다.
        returns = np.dot(weights,annual_ret)

        # 종목별 연간 공분산과 종목별 비중 배열을 곱한뒤 이를 다시 종목별 비중의 전치로 곱한다.
        # 이렇게 구한 결과값의 제곱근 sqrt() 구하면 해당 포트폴리오의 전체 리스크를 구할수 있다.
        risk = np.sqrt(np.dot(weights.T,np.dot(annual_cov,weights)))

        port_ret.append(returns)
        port_risk.append(risk)
        port_weights.append(weights)
        sharp_ratio.append(returns/risk) #샤프지수

    portfolio = {'Returns':port_ret, 'Risk':port_risk, 'Sharp':sharp_ratio}

    for i,s in enumerate(stock): #각 종목별로 비중 입력
        portfolio[s]=[weights[i] for weights in port_weights]

    df = pd.DataFrame(portfolio)
    df = df[['Returns','Risk','Sharp'] + [s for s in stock]]

    max_sharp = df.loc[df['Sharp'] == df['Sharp'].max()]
    min_risk = df.loc[df['Risk'] == df['Risk'].min()]

    df.plot.scatter(x='Risk',y='Returns',c='Sharp',cmap='viridis',edgecolor='k',figsize=(11,7),grid=True)
    plt.scatter(x=max_sharp['Risk'],y=max_sharp['Returns'],c='r',marker='*',s=300) #샤프지수가 가장높은곳
    plt.scatter(x=min_risk['Risk'],y=min_risk['Returns'],c='r',marker='X',s=200) #risk가 가장 낮은 곳
    plt.title('Portfolio Optimization')
    plt.xlabel('Risk')
    plt.ylabel('Expected Returns')
    figure = plt.show()
    st.pyplot(figure)

    st.write("Max Sharp Portfolio",max_sharp)
    st.write("Min Risk Portfolio",min_risk)
    
def BollingerBandAnalysis():
    
    st.subheader("John Bollinger - 볼린저 밴드 투자기법 ")

    #볼린저 밴드 설명
    st.write(":moneybag:표준 볼린저 밴드 공식")

    st.write("상단 볼린저 밴드 = 중간볼린저밴드 + (2 x 표준편차)")
    st.write("상단 볼린저 밴드 = 종가의 20일 이동평균")
    st.write("상단 볼린저 밴드 = 중간볼린저밴드 - (2 x 표준편차)")

    st.write("주가가 볼린저 밴드 어디에 위치하는지를 나타내는 지료가 %b다. 상단밴드에 걸쳐있을때 1.0, 중가 0.5 하단 0.0")
    st.write(":green[$ \%b = \dfrac{종가 - 하단 볼린저 밴드} {상단 볼린저 밴드 - 하단 볼린저 밴드}$]")

    st.write(":moneybag:밴드폭은 상단 볼린저 밴드와 하단 볼린저 밴드 사이의 폭을 의미한다.")
    st.write("밴드폭은 스퀴즈를 확인하는데 유용한 지표이다. 스퀴즈란 변동성이 극히 낮은 수준까지 떨어져 곧이어"
             "변동성 증가가 발생할 것으로 예상되는 상황을 말한다. 볼린저가 저술한바에 따르면 밴드폭이"
             "6개월 저점을 기록하는 것을 보고 스퀴즈를 파악할수 있다고 한다.")
    st.write("밴드폭 산출 공식")
    st.write(":green[$ 밴드폭 = \dfrac{상단 볼린저 밴드 - 하단 볼린저 밴드} {중간 볼린저 밴드}$]")
    st.write("밴드폭은 또 다른 중요한 역활은 강력한 추세의 시작과 마지막을 포착하는 것이다."
             "강력한 추세는 스퀴즈로부터 시작되는데 변동성이 커지면서 밴드폭 수치가 급격히 높아진다."
             "이때 밴드폭이 넓어지면서 추세의 반대쪽에 있는 밴드는 추세 반대 방향으로 향한다.")

    s1 = st.text_input("분석종목: ", value='AAPL')
    date = st.date_input("시작날짜", datetime.date(2022, 6, 1))

    st.subheader("추세 추종 매매기법")
    st.write("추세 추종은 상승추세에 매수하고 하락추세에 매도하는 기법")
    
    #실제 코드
    f1 = pdr.get_data_yahoo(s1,date)

    f1.index = f1.index.date #index 시간제거

    df = pd.DataFrame(f1)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['stddev'] = df['Close'].rolling(window=20).std()
    df['upper'] = df['MA20'] + (df['stddev'] * 2)
    df['lower'] = df['MA20'] - (df['stddev'] * 2)
    df['bandwidth'] = (df['upper'] - df['lower'] ) / df['MA20'] * 100
    df['PB'] = (df['Close']- df['lower']) / (df['upper'] - df['lower'])
    df = df[19:] #19번째까지 Nan이므로 삭제

    # st.write(df)
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['PMF'] = 0
    df['NMF'] = 0

    for i in range(len(df.Close) - 1):
        if df.TP.values[i] < df.TP.values[i + 1]:  # i+1번째 가격이 i보다 높으면
            df.PMF.values[i + 1] = df.TP.values[i + 1] * df.Volume.values[i + 1]  # 긍정적 현금흐름에 저장
            df.NMF.values[i + 1] = 0
        else:
            df.NMF.values[i + 1] = df.TP.values[i + 1] * df.Volume.values[i + 1]  # 아니면 부정적 현금흐름에 저장
            df.PMF.values[i + 1] = 0

    df['MFR'] = df.PMF.rolling(window=10).sum() / df.NMF.rolling(window=10).sum()
    df['MFI10'] = 100 - 100 / (1 + df['MFR'])


    plt.figure(figsize=(9,10))
    plt.subplot(3,1,1)
    plt.plot(df.index,df['Close'],color='#0000ff',label='Close')
    plt.plot(df.index,df['upper'],'r--',label='Upper Band')
    plt.plot(df.index,df['MA20'],'k--',label='Moving average 20')
    plt.plot(df.index,df['lower'],'c--',label='Lower Band')
    plt.fill_between(df.index,df['upper'],df['lower'],color='0.9')
    for i in range(len(df.Close)):
        if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
            plt.plot(df.index.values[i],df.Close.values[i],'r^')
        elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
            plt.plot(df.index.values[i],df.Close.values[i],'bv')
    plt.legend(loc='best')
    plt.title(f"{s1} Bollinger Band (20 day, 2std)")

    plt.subplot(3,1,2)
    plt.plot(df.index,df['PB']*100,color='b',label='%B x 100')
    plt.plot(df.index,df['MFI10'],'g--',label='MFI(10 days)')
    plt.yticks([-20,0,20,40,60,80,100,120]) #y축을 20단위로 표시
    for i in range(len(df.Close)):
        if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
            plt.plot(df.index.values[i],0,'r^')
        elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
            plt.plot(df.index.values[i],0,'bv')
    plt.grid(True)
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    plt.plot(df.index,df['bandwidth'],color='m',label='BandWidth')
    plt.grid(True)
    plt.legend(loc='best')

    figure = plt.show()
    st.pyplot(figure)



    #추세추정 매매기법설명

    st.write("상승 추세나 하락 추세의 시작을 단순히 %b 지표만 이용해서 주가가 볼린저 상/하단 밴드에 태그했는지여부"
             "로만 판단하지 않는다. 현금흐름지표(MFI)나 일중강도(II) 같은 거래량 관련 지표를 함께 이용해서 확증이"
             "이루어진 경우에만 매수/매도에 들아간다.")
    st.write(":red[매수 : 주가가 상단밴드에 접근하며, 지표가 강세를 확증할 때만 매수 (%b가 0.8보다 크고, MFI가 80보다 클때)]")
    st.write(":red[매수 : 주가가 하단밴드에 접근하며, 지표가 약세를 확증할 때만 매도(%b가 0.2보다 작고, MFI가 20보다 작을때)]")

    st.write(":moneybag: Money Flow Index 현금흐름지표 : 가격과 거래량을 동시에 분석하므로 상대적으로 신뢰도가 더 높다.")
    st.write("중심가격 = (일정기간의 고가+저가+종가) / 3")
    st.write("현금흐름 = 중심가격 * 거래량")

    st.write(":green[$ MFI = 100 - \Bigg( 100 \\div ( 1 + \dfrac{긍정적 현금흐름} {부정적 현금흐름}) \Bigg) $]")
    st.write(":green[긍정적 현금흐름 = 중심가격이 전일보다 상승한 날들의 현금 흐름의 합]")
    st.write(":green[부정적 현금흐름 = 중심가격이 전일보다 하락한 날들의 현금 흐름의 합]")

    st.subheader("반전 매매기법")
    st.write("주가가 반전하는 지점을 찾아내 매수/매도하는 기법")

    # 위에 코드 이어서 작성
    df['II'] = (2*df['Close'] - df['High'] - df['Low']) / (df['High']-df['Low']) * df['Volume'] #일중강도
    df['IIP21'] = df['II'].rolling(window=21).sum() / df['Volume'].rolling(window=21).sum()*100 #일중강도율
    df = df.dropna()

    plt.figure(figsize=(9,10))
    plt.subplot(3,1,1)
    plt.title(f"{s1} Bollinger Band (20 day, 2std) - Reversals")
    plt.plot(df.index,df['Close'],color='#0000ff',label='Close')
    plt.plot(df.index,df['upper'],'r--',label='Upper Band')
    plt.plot(df.index,df['MA20'],'k--',label='Moving average 20')
    plt.plot(df.index,df['lower'],'c--',label='Lower Band')
    plt.fill_between(df.index,df['upper'],df['lower'],color='0.9')
    for i in range(0,len(df.Close)):
        if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
            plt.plot(df.index.values[i],df.Close.values[i],'r^')
        elif df.PB.values[i] > 0.95  and df.IIP21.values[i] < 0:
            plt.plot(df.index.values[i],df.Close.values[i],'bv')
    plt.legend(loc='best')

    plt.subplot(3,1,2)
    plt.plot(df.index,df['PB'],'b',label='%b')
    plt.grid(True)
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    plt.bar(df.index,df['IIP21'],color='g',label='II% 21day')
    for i in range(len(df.Close)):
        if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
            plt.plot(df.index.values[i],0,'r^')
        elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:
            plt.plot(df.index.values[i],0,'bv')
    plt.grid(True)
    plt.legend(loc='best')

    figure = plt.show()
    st.pyplot(figure)

    #반전매매기법 설명

    st.write("주가가 하단 밴드를 여러 차례 태그하는 과정에서 강세 지표가 발생하면 매수하고, 주가가 상단 밴드를 여러차례 태그하는 과정에서"
             "약세 지표가 발생하면 매도한다.")

    st.write(":red[매수 : 주가가 하단 밴드부근에서 W형 패턴을 나타내고, 강세 지표가 확증할 때 매수(%b가 0.05보다 작고 일중강도율II%가 0보다 크면 매수]")
    st.write(":red[매수 : 주가가 상단 밴드부근에서 일련의 주가태그가 일어나며, 약세 지표가 확증할 때 매도(%b가 0.95보다 크고 일중강도율II%가 0보다 작으면 매도]")
    st.write(":moneybag: 일중강도는 데이빗 보스티언이 개발한 거래량지표 : 거래 범위에서 종가의 위치를 토대로 주식 종목의 "
             "자금 흐름을 설명한다. 장이 끝나는 시점에서 트레이더들의 움직임을 나타내는데, 종가가 거래범위 천정권에서 형성되면 1,"
             "중간에서 형성되면 0, 바닥권에서 형성되면 -1이 된다.")
    st.write("21일 기간동안 II합을 21 기간동안의 거래량 합으로 나누어 표준화한 것이 일중 강도율II%이다.")
    st.write(":green[$ 일중강도 = \dfrac{2 * 종가 - 고가 - 저가} {고가 - 저가} * 거래량  $]")
    st.write(":green[$ 일중강도율 = \dfrac{일중강도의 21일 합} {거래량의 21일 합} * 100  $]")

def TradingforaLiving():

    st.subheader("알렉산더 엘더 - 주식시장에서 살아남는 심리투자 법칙")
    st.write("삼중창 매매 시스템 - 추세 추종과 역추세 매매법을 함께 사용하며, 세 단계의 참을 거쳐 더 정확한 매매 시점을 찾도록 구성되어 있다.")
    st.write(":moneybag:첫번째 창")
    st.write("시장의 장기 추세를 분석하기 위해 26주 지수 이동평균에 해당하는 EMA130그래프와 주간 MACD히스토그램을 함께 표시")

    s1 = st.text_input("분석종목: ", value='SOXL')
    date = st.date_input("시작날짜", datetime.date(2020, 1, 2))

    # 실제 코드
    f1 = pdr.get_data_yahoo(s1, date)

    f1.index = f1.index.date  # index 시간제거
    df = pd.DataFrame(f1)

    ema60 = df.Close.ewm(span=60).mean() #종가의 12주 지수 이동평균
    ema130 = df.Close.ewm(span=130).mean() #종가의 26주 지수 이동평균
    macd = ema60 - ema130 #macd 선
    signal = macd.ewm(span=45).mean() #신호선 (macd의 9주 지수 이동평균)
    macdhist = macd - signal

    df = df.assign(ema130=ema130, ema60=ema60,macd=macd, signal=signal,macdhist=macdhist).dropna()
    df['number'] = df.index.map(mdates.date2num) #캔들차트에 사용할수 있게 날짜형 인덱스를 숫자형으로 변환
    ohlc = df[['number','Open','High','Low','Close']]

    plt.figure(figsize=(9,7))

    p1 = plt.subplot(2,1,1)
    plt.title(f'Triple Screen Trading - First Screen {s1}')
    plt.grid(True)
    candlestick_ohlc(p1,ohlc.values,width=.6,colorup='red',colordown='blue') #ohlc의 숫자형 일자,시가,고가,저가,종가 값을 이용해서 캔들차트를 그린다.
    p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot(df.number,df['ema130'],color='c',label='EMA130')
    plt.legend(loc='best')

    p2=plt.subplot(2,1,2)
    plt.grid(True)
    p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.bar(df.number,df['macdhist'],color='m',label='MACD-Hist')
    plt.plot(df.number,df['macd'],color='b',label='MACD')
    plt.plot(df.number,df['signal'],'g--',label='MACD-signal')
    plt.legend(loc='best')
    figure = plt.show()
    st.pyplot(figure)


