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




def MACDStrategy():
    """
    macd Oscillator 의 기울기로 매수 매도 비중을 조절한다.
    :return:
    """
    st.subheader("MACD, MACD Oscillator, RSI 활용한 주식투자 by 502")

    selected = st.checkbox("보유종목")
    date = st.date_input("시작날짜", datetime.date(2022, 10, 1))
    if selected:
        list = ['AAPL', 'ABBV', 'GOOGL', 'INTC', 'NAIL', 'SOXL', 'SOXS']

        # 실제 코드
        for s in list:
            st.subheader(f"분석 종목 : {s}")
            f1 = pdr.get_data_yahoo(s, date)

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
            plt.title(f'MACD / MACD Osillator / RSI Trading {s}')
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
                if df.macdhist.values[i - 1] < df.macdhist.values[i] and df.rsi.values[i]<45 :
                    plt.plot(df.number.values[i], 0, 'r^')
                elif df.macdhist.values[i - 1] > df.macdhist.values[i] and df.rsi.values[i]>60 :
                    plt.plot(df.number.values[i], 0, 'bv')

            plt.legend(loc='best')

            p2 = plt.subplot(3, 1, 3)
            plt.grid(True)
            p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.plot(df.number, df['rsi'], color='b', label='%RSI')
            plt.axhline(60,0,1,color = 'green', linestyle ='--')
            plt.axhline(45,0,1,color = 'green', linestyle ='--')
            plt.yticks([20,30,40,50,60,70, 80])
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

            # cerebro.plot()
            #
            # figure = cerebro.plot(style='candlestick')[0][0] #캔들차트로 설정
            #
            # # show the plot in Streamlit
            # st.pyplot(figure)

    else:
        stock = st.text_input("분석종목: ", value='SOXL')

        f1 = pdr.get_data_yahoo(stock, date)

        day_list = []
        name_list = []

        for i, day in enumerate(f1.index):
            if day.dayofweek == 0:
                day_list.append(i)
                name_list.append(day.strftime('%Y-%m-%d'))

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
        plt.title(f'MACD / MACD Osillator / RSI Trading {stock}')
        plt.grid(True)

        #p1.xaxis.set_major_locator(ticker.FixedLocator(day_list))
        #p1.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))
        candlestick_ohlc(p1, ohlc.values, width=.5, colorup='red',
                         colordown='blue')  # ohlc의 숫자형 일자,시가,고가,저가,종가 값을 이용해서 캔들차트를 그린다.

        p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(df.number, df['ema26'], color='c', label='EMA26')
        plt.plot(df.number, df['ema12'], color='g', label='EMA12', linestyle='--')
        plt.legend(loc='best')

        p2 = plt.subplot(3, 1, 2)
        plt.grid(True)
        #p2.xaxis.set_major_locator(ticker.FixedLocator(day_list))
        #p2.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))
        p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.bar(df.number, df['macdhist'], color='m', label='MACD-Hist')
        plt.plot(df.number, df['macd'], color='b', label='MACD')
        plt.plot(df.number, df['signal'], 'g--', label='MACD-signal') \

        for i in range(1, len(df.Close)):
            if df.macdhist.values[i - 1] < df.macdhist.values[i] and df.rsi.values[i] < 45:
                plt.plot(df.number.values[i], 0, 'r^')
            elif df.macdhist.values[i - 1] > df.macdhist.values[i] and df.rsi.values[i] > 60:
                plt.plot(df.number.values[i], 0, 'bv')

        plt.legend(loc='best')

        p2 = plt.subplot(3, 1, 3)
        plt.grid(True)
        #p2.xaxis.set_major_locator(ticker.FixedLocator(day_list))
        #p2.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))
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

        feed = bt.feeds.PandasData(dataname=yf.download(stock, date))

        # st.write(data)
        cerebro.adddata(feed)  # Add the data feed
        cerebro.broker.setcash(10000000)
        cerebro.broker.setcommission(commission=0.0014)

        cerebro.addsizer(bt.sizers.PercentSizer, percents=90)  # 매매 단위
        #cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='areturn')
        st.write(f'Initial Porfolio Value : {cerebro.broker.getvalue():,.0f} USD')
        cerebro.run()  # run it all

        st.write(f'Final Porfolio Value : {cerebro.broker.getvalue():,.0f} USD')

        cerebro.plot()
        #
        figure = cerebro.plot(style='candlestick')[0][0] #캔들차트로 설정
        #
        # # show the plot in Streamlit
        st.pyplot(figure)