
# import pandas_ta as ta
#TA-Lib install StreamLit
import requests
import os
import sys
import subprocess

# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    os.system("ls -la /app/equity/")
    # build
    os.system("./configure --prefix=/home/appuser")

    # os.system('export TA_INCLUDE_PATH = "/home/appuser/include"')
    # os.system('export TA_INCLUDE_PATH = "/home/appuser/lib"')

    os.system("make")
    # install
    os.system("make install")

    # back to the cwd
    os.chdir(default_cwd)
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
# import library
try:
    import talib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/", "--global-option=-I/home/appuser/include/", "ta-lib"])
finally:
    import talib

#TA-Lib install StreamLit

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
