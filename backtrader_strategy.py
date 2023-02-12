from datetime import datetime
import streamlit as st
import backtrader as bt
import backtrader.feeds as btfeeds
# from matplotlib.dates import warnings
import FinanceDataReader as fdr
import pandas as pd
import matplotlib
import yfinance as yf
yf.pdr_override()


class Mystrategy(bt.Strategy):

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period =21) #RSI_SMA 지표 사용

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
            if self.rsi < 30: #rsi 30보다 낮으면 매수
                self.order = self.buy()
        else:
            if self.rsi > 70: #rsi 30보다 높으면 매도
                self.order = self.sell()

    def log(self,txt,dt=None): #텍스트 인수를 전달 받아서 셀화면에 주문일자와 함께 출력하는 역활을 한다.
        dt = self.datas[0].datetime.date(0)
        st.write(f'[{dt.isoformat()}] {txt}')


class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30  # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position


def Backtrader():

    # Use a backend that doesn't display the plot to the user
    # we want only to display inside the Streamlit page
    matplotlib.use('Agg')

    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
    cerebro.addstrategy(Mystrategy)  # Add the trading strategy
    # Create a data feed
    selected_stock_value = st.text_input('Input STOCK CODE to use in backtest', value='AAPL')

    start_date = st.date_input("Choice a Start Day",datetime(2017, 1, 2))
    end_date = st.date_input("Choice a Start Day",datetime(2019, 12, 30))
    # data = fdr.DataReader(selected_stock_value, start_date,end_date)
    # feed = bt.feeds.PandasData(dataname=data)
    feed = bt.feeds.PandasData(dataname=yf.download(selected_stock_value, start_date, end_date))
    # data = bt.feeds.YahooFinanceData(dataname='036570.KS',
    #                                   fromdate=datetime(2017, 1, 1),
    #                                   todate=datetime(2019, 12, 31))

    # st.write(data)
    cerebro.adddata(feed)  # Add the data feed
    cerebro.broker.setcash(10000000)
    cerebro.broker.setcommission(commission=0.0014)

    cerebro.addsizer(bt.sizers.PercentSizer,percents=90) #매매 단위를 30주로 셋팅

    st.write(f'Initial Porfolio Value : {cerebro.broker.getvalue():,.0f} USD')
    cerebro.run()  # run it all

    st.write(f'Final Porfolio Value : {cerebro.broker.getvalue():,.0f} USD')
    # cerebro.plot()

    # figure = cerebro.plot(style='candlestick')[0][0] #캔들차트로 설정
    #
    # # show the plot in Streamlit
    # st.pyplot(figure)

