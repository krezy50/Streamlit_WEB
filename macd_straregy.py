import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import yfinance as yf
import backtrader as bt
import datetime

from mplfinance.original_flavor import candlestick_ohlc
from pandas_datareader import data as pdr


yf.pdr_override()


class Mystrategy(bt.Strategy):

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period =21) #RSI_SMA 지표 사용
        self.macdhisto = bt.indicators.MACDHisto(self.data.close)

    def notify_order(self, order): #order stats가 변할때 자동적으로 실행
        if order.status in [order.Submitted,order.Accepted]:
            return

        if order.status in [order.Completed]: #주문상태가 Completed 되면 상세 주문정보를 출력
            if order.isbuy():
                self.log(f'BUY : 주가 {order.executed.price:,.0f},'
                         f'수량 {order.executed.size:.0f},'
                         f'수수료 {order.executed.comm:.0f},'
                         # f'자산 {cerebro.broker.getvalue():.0f},'
                         )
            else:
                self.log(f'SELL : 주가 {order.executed.price:,.0f},'
                         f'수량 {order.executed.size:.0f},'
                         f'수수료 {order.executed.comm:.0f},'
                         # f'자산 {cerebro.broker.getvalue():.0f},'
                         )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled]:
            self.log('ORDER CANCELD')
        elif order.status in [order.Margin]:
            self.log('ORDER MARGIN')
        elif order.status in [order.Rejected]:
            self.log('ORDER REJECTED')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(['매매수익률', '{:.0f}'.format(trade.pnl)])

    def next(self): #주어진 데이터와 지표를 만족시키는 최소 주기마다 자동으로 호출된다.
        if not self.position:
            if self.rsi < 45 and self.macdhisto[-1] < self.macdhisto[0]  : #rsi 30보다 낮으면 매수
                self.order = self.buy()
        else:
            if self.rsi > 60 and self.macdhisto[-1] > self.macdhisto[0] : #rsi 30보다 높으면 매도
                self.order = self.sell()

    def log(self,txt,dt=None): #텍스트 인수를 전달 받아서 셀화면에 주문일자와 함께 출력하는 역활을 한다.
        dt = self.datas[0].datetime.date(0)
        st.write(f'[{dt.isoformat()}] {txt}')

def macd_rsi(s, date): # 전략 version 0.1

    st.subheader(f"분석 종목 : {s}")
    f1 = pdr.get_data_yahoo(s, date)

    f1.index = f1.index.date  # index 시간제거
    df = pd.DataFrame(f1)

    ema12 = df.Close.ewm(span=12).mean()  # 종가의 12일 지수 이동평균
    ema26 = df.Close.ewm(span=26).mean()  # 종가의 26일 지수 이동평균
    macd = ema12 - ema26  # macd 선
    signal = macd.ewm(span=9).mean()  # 신호선 (macd의 9일 지수 이동평균)
    macdhist = macd - signal

    period = 14
    delta = df.Close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    gain = up.ewm(com=(period - 1), min_periods=period).mean()
    loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()
    RS = gain / loss
    rsi = 100 - (100 / (1 + RS))

    df = df.assign(ema26=ema26, ema12=ema12, macd=macd, signal=signal, macdhist=macdhist, rsi=rsi).dropna()

    df['number'] = df.index.map(mdates.date2num)  # 캔들차트에 사용할수 있게 날짜형 인덱스를 숫자형으로 변환
    ohlc = df[['number', 'Open', 'High', 'Low', 'Close']]

    plt.figure(figsize=(9, 9))

    p1 = plt.subplot(3, 1, 1)
    plt.title(f'MACD / MACD Osillator / RSI Trading {s}')
    plt.grid(True)
    candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red',
                     colordown='blue')  # ohlc의 숫자형 일자,시가,고가,저가,종가 값을 이용해서 캔들차트를 그린다.
    p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot(df.number, df['ema26'], color='c', label='EMA26')
    plt.plot(df.number, df['ema12'], color='g', label='EMA12', linestyle='--')
    plt.legend(loc='best')

    p2 = plt.subplot(3, 1, 2)

    plt.grid(True)
    p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.bar(df.number, df['macdhist'], color='m', label='MACD-Hist')
    plt.plot(df.number, df['macd'], color='b', label='MACD')
    plt.plot(df.number, df['signal'], 'g--', label='MACD-signal')

    for i in range(1, len(df.Close)):
        if df.macdhist.values[i - 1] < df.macdhist.values[i] and df.rsi.values[i] < 45:
            plt.plot(df.number.values[i], 0, 'r^')
        elif df.macdhist.values[i - 1] > df.macdhist.values[i] and df.rsi.values[i] > 60:
            plt.plot(df.number.values[i], 0, 'bv')

    plt.legend(loc='best')

    p2 = plt.subplot(3, 1, 3)
    plt.grid(True)
    p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot(df.number, df['rsi'], color='b', label='%RSI')
    plt.axhline(60, 0, 1, color='green', linestyle='--')
    plt.axhline(45, 0, 1, color='green', linestyle='--')
    plt.yticks([20, 30, 40, 50, 60, 70, 80])
    plt.legend(loc='best')
    figure = plt.show()
    st.pyplot(figure)

    st.subheader("BackTest 결과")
    # Use a backend that doesn't display the plot to the user
    # we want only to display inside the Streamlit page
    matplotlib.use('Agg')

    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
    cerebro.addstrategy(Mystrategy)  # Add the trading strategy

    feed = bt.feeds.PandasData(dataname=yf.download(s, date))

    # st.write(data)
    cerebro.adddata(feed)  # Add the data feed
    cerebro.broker.setcash(10000000)
    cerebro.broker.setcommission(commission=0.0014)

    cerebro.addsizer(bt.sizers.PercentSizer, percents=90)  # 매매 단위

    st.write(f'Initial Porfolio Value : {cerebro.broker.getvalue():,.0f} USD')
    cerebro.run()  # run it all

    st.write(f'Final Porfolio Value : {cerebro.broker.getvalue():,.0f} USD')

def macd_rsi_variation(s, date): # 전략 version 0.2

    st.subheader(f"분석 종목 : {s}")
    f1 = pdr.get_data_yahoo(s, date)

    # f1.index = f1.index.date  # index 시간제거
    df = pd.DataFrame(f1)

    ema12 = df.Close.ewm(span=12).mean()  # 종가의 12일 지수 이동평균
    ema26 = df.Close.ewm(span=26).mean()  # 종가의 26일 지수 이동평균
    macd = ema12 - ema26  # macd 선
    signal = macd.ewm(span=9).mean()  # 신호선 (macd의 9일 지수 이동평균)
    macdhist = macd - signal

    period = 14
    delta = df.Close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    gain = up.ewm(com=(period - 1), min_periods=period).mean()
    loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()
    RS = gain / loss
    rsi = 100 - (100 / (1 + RS))

    df = df.assign(ema26=ema26, ema12=ema12, macd=macd, signal=signal, macdhist=macdhist, rsi=rsi).dropna()

    RSI_THRESHOLD = 0.20 #rsi 평균에 20%
    rsi_ewm = df.rsi.ewm(span=52).mean() #rsi 지난 2개월 평균값
    rsi_max_threshold = rsi_ewm + (rsi_ewm * RSI_THRESHOLD)
    rsi_min_threshold = rsi_ewm - (rsi_ewm * RSI_THRESHOLD)

    df = df.assign(rsi_ewm=rsi_ewm,rsi_max_threshold=rsi_max_threshold,rsi_min_threshold=rsi_min_threshold).dropna()

    df['number'] = df.index.map(mdates.date2num)  # 캔들차트에 사용할수 있게 날짜형 인덱스를 숫자형으로 변환
    ohlc = df[['number', 'Open', 'High', 'Low', 'Close']]

    plt.figure(figsize=(9, 9))

    p1 = plt.subplot(3, 1, 1)
    plt.title(f'MACD / MACD Osillator / RSI Trading {s}')
    plt.grid(True)
    candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red',
                     colordown='blue')  # ohlc의 숫자형 일자,시가,고가,저가,종가 값을 이용해서 캔들차트를 그린다.
    p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot(df.number, df['ema26'], color='g', label='EMA26')
    plt.plot(df.number, df['ema12'], color='g', label='EMA12', linestyle='--')
    plt.legend(loc='best')

    p2 = plt.subplot(3, 1, 2)

    plt.grid(True)
    p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.bar(df.number, df['macdhist'], color='m', label='MACD-Hist')
    plt.plot(df.number, df['macd'], color='b', label='MACD')
    plt.plot(df.number, df['signal'], 'b--', label='MACD-signal')

    for i in range(1, len(df.Close)):
        if df.macdhist.values[i - 1] > df.macdhist.values[i] and df.rsi.values[i - 1] > df.rsi_max_threshold.values[i - 2] :
            plt.plot(df.number.values[i], 0, 'bv')
        # elif df.macd.values[i - 1] > df.signal.values[i - 1] and df.macd.values[i] < df.signal.values[i] :
        #     plt.plot(df.number.values[i], 0, 'bv')

        elif df.macdhist.values[i - 1] < df.macdhist.values[i] and df.rsi.values[i - 1] < df.rsi_min_threshold.values[i - 2] :
            plt.plot(df.number.values[i], 0, 'r^')
        # elif df.macd.values[i - 1] < df.signal.values[i - 1] and df.macd.values[i] > df.signal.values[i] :
        #     plt.plot(df.number.values[i], 0, 'r^')


    plt.legend(loc='best')

    p2 = plt.subplot(3, 1, 3)
    plt.grid(True)
    p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot(df.number, df['rsi'], color='b', label='RSI')
    plt.plot(df.number, df['rsi_ewm'], color='r', label='RSI_ewm')
    plt.plot(df.number, df['rsi_max_threshold'], 'g--', label='RSI_max')
    plt.plot(df.number, df['rsi_min_threshold'], 'g--', label='RSI_min')
    plt.fill_between(df.number, df['rsi_max_threshold'], df['rsi_min_threshold'], color='0.9')
    # plt.axhline(df.rsi_ewm.values[i]+RSI_THRESHOLD, 0, 1, color='green', linestyle='--')
    # plt.axhline(df.rsi_ewm.values[i]-RSI_THRESHOLD, 0, 1, color='green', linestyle='--')
    plt.yticks([20, 30, 40, 50, 60, 70, 80])
    plt.legend(loc='best')
    figure = plt.show()
    st.pyplot(figure)

    for i in range(1, len(df.Close)):
        if df.macdhist.values[i - 1] > df.macdhist.values[i] and df.rsi.values[i - 1] > df.rsi_max_threshold.values[i - 2] :
            sell = df.index.values[i]
            sell = pd.to_datetime(sell)
            st.write(f":blue[:chart_with_downwards_trend:매도 신호 : {format(sell.date())}, 매도가 : {round(df.Close.values[i],2)}]")

        elif df.macdhist.values[i - 1] < df.macdhist.values[i] and df.rsi.values[i - 1] < df.rsi_min_threshold.values[i - 2] :

            buy = df.index.values[i]
            buy = pd.to_datetime(buy)
            st.write(f":red[:chart_with_upwards_trend:매수 신호 : {format(buy.date())}, 매수가 : {round(df.Close.values[i],2)}]")


    # st.subheader("BackTest 결과")
    # # Use a backend that doesn't display the plot to the user
    # # we want only to display inside the Streamlit page
    # matplotlib.use('Agg')
    #
    # cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
    # cerebro.addstrategy(Mystrategy)  # Add the trading strategy
    #
    # feed = bt.feeds.PandasData(dataname=yf.download(s, date))
    #
    # # st.write(data)
    # cerebro.adddata(feed)  # Add the data feed
    # cerebro.broker.setcash(10000000)
    # cerebro.broker.setcommission(commission=0.0014)
    #
    # cerebro.addsizer(bt.sizers.PercentSizer, percents=90)  # 매매 단위
    #
    # st.write(f'Initial Porfolio Value : {cerebro.broker.getvalue():,.0f} USD')
    # cerebro.run()  # run it all
    #
    # st.write(f'Final Porfolio Value : {cerebro.broker.getvalue():,.0f} USD')


def MACDStrategy():
    """
    macd Oscillator 의 기울기로 매수 매도 비중을 조절한다.
    :return:
    """
    st.subheader("MACD, MACD Oscillator, RSI 활용한 주식투자 v0.02 by 502")
    st.write("해당 전략은 참고용이며, 모든 투자는 투자자에게 책임이 있습니다.")
    st.write(":moneybag:종목 과매수/과매도 기준 : RSI 52일 평균지수의 +20% max, -20% min  설정")
    st.caption("(RSI max, min 값을 고정하지 않고 종목마다 상대적으로 설정)")
    st.write(":red[:chart_with_upwards_trend:매수 조건 : MACD Hist 이 전날보다 높으면서, 전날 RSI가 그 전날 RSI min보다 낮을 경우 매수]")
    st.caption("종목 과매도 구간이 진정되는 시점에 매수 신호 발생, 매수는 분할 매수로 진행")
    st.write(":blue[:chart_with_downwards_trend:매도 조건 : MACD Hist 이 전날보다 낮으면서, 전날 RSI가 그 전날 RSI max보다 높을 경우 매도]")
    st.caption("종목 과매수 구간이 진정되는 시점에 매도 신호 발생, 매도는 분할로 하되 보유 금액의 1/3 씩 진행, 수익률이 10% 이상 일 경우 1/3은 지속 보유")

    
    selected = st.checkbox("보유종목 분석")
    date = st.date_input("시작날짜", datetime.datetime.today()-datetime.timedelta(days=180)) #6개월

    if selected:
        list = ['AAPL', 'AMZN', 'GOOGL', 'INTC', 'NAIL', 'SOXL', 'TSM']

        # 실제 코드
        for s in list:

            macd_rsi_variation(s, date)

    else:
        stock = st.text_input("분석종목: ", value='SOXL')

        # macd_rsi(stock,date)
        macd_rsi_variation(stock,date)
