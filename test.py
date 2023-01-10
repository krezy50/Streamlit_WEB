# import pandas as pd
# import streamlit as st
# from backtesting import Backtest, Strategy
# from backtesting.lib import crossover
# import FinanceDataReader as fdr
#
# from ta.trend import MACD
# import ta
#
# # import pandas_ta as ta
# import datetime
#
# start_date = datetime.date(2015, 1, 1)
# price_df = fdr.DataReader('SOXL', start_date)
#
# # macd = ta.macd(price_df['Close'])
# # st.write(macd)
#
# def MACD_cross(close, n1, n2, ns):
#
#     # df = ta.macd(close,  fast=n1, slow=n2, signal=ns,asmode=True,fillna=False)
#     # st.write(df)
#     #
#     # df.rename(columns={f'MACD_{n1}_{n2}_{ns}':'macd',f'MACDh_{n1}_{n2}_{ns}':'macdh',f'MACDs_{n1}_{n2}_{ns}':'macds'},inplace=True)
#     #
#     # macd = df[f'MACD_{n1}_{n2}_{ns}']
#     # signalma = df[f'MACDs_{n1}_{n2}_{ns}']
#     #
#     macd = ta.trend.macd(close,  window_fast=n1, window_slow =n2)
#     signalma = ta.trend.macd_signal(close,  window_fast=n1, window_slow =n2, window_sign =ns)
#
#     return macd[0],signalma[0]
#
# class MACDCross(Strategy):
#
#     short_term = 12
#     long_term = 26
#     sequence = 9
#
#     def init(self):
#         close = self.data.Close
#         self.macd, self.signalma = self.I(MACD_cross,close, self.short_term, self.long_term, self.sequence)
#
#     def next(self):
#         if crossover(self.macd, self.signalma):
#             self.buy()
#         elif crossover(self.signalma, self.macd):
#             self.position.close()
#
# bt = Backtest(price_df, MACDCross,
#               cash=10000, commission=.002,
#               exclusive_orders=True)
#
# output = bt.run()
# df = pd.DataFrame(output)
# st.write(df)
