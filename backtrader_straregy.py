from datetime import datetime
import streamlit as st
import backtrader as bt
import backtrader.feeds as btfeeds
import FinanceDataReader as fdr
import pandas as pd
import matplotlib


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

    # Create a data feed
    selected_stock_value = st.text_input('Input STOCK CODE to use in backtest', value='SOXL')

    start_date = st.date_input("Choice a Start Day",datetime(2015, 1, 1))
    data = fdr.DataReader(selected_stock_value, start_date)
    feed = bt.feeds.PandasData(dataname=data)
    # data1 = bt.feeds.YahooFinanceData(dataname='AAPL',
    #                                   fromdate=datetime(2011, 1, 1),
    #                                   todate=datetime(2021, 12, 31))

    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
    cerebro.adddata(feed)  # Add the data feed

    cerebro.addstrategy(SmaCross)  # Add the trading strategy
    cerebro.run()  # run it all
    # cerebro.plot()

    figure = cerebro.plot()[0][0]

    # show the plot in Streamlit
    st.pyplot(figure)

