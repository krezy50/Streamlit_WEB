# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import pandas as pd

from rental_investment_calculator import RentalInvestmentCalculator #임대 수익 계산기
from market_rate import MarketRateScrapping #스크랩핑
from control_excel import convert_df



# import FinanceDataReader as fdr
# import pandas as pd
# from backtesting import Backtest, Strategy
# from backtesting.lib import crossover
#
# from backtesting.test import SMA, GOOG
#
# class SmaCross(Strategy):
#     def init(self):
#         price = self.data.Close
#         self.ma1 = self.I(SMA, price, 10)
#         self.ma2 = self.I(SMA, price, 20)
#
#     def next(self):
#         if crossover(self.ma1, self.ma2):
#             self.buy()
#         elif crossover(self.ma2, self.ma1):
#             self.sell()

#메인 : 시스템 선택

with st.form("시스템 선택"):
    st.header("Python projects of 502")

    system=st.radio("Choice a project", ('임대 수익률 계산기', '시장 금리 스크래핑','BackTesing'))
    submitted = st.form_submit_button("Submit")

if system == '임대 수익률 계산기':
    st.caption("Sidebar MENU 에서 관련 정보를 입력하세요.")
    result = RentalInvestmentCalculator()


    title = st.text_input("다운받을 파일명을 입력하세요.",max_chars=10)
    add_title = result.rename(columns={0:title}) #제목 index으로 변경
    csv = convert_df(add_title)

    st.download_button( #파일에 쓰고 다운로드
        label="Download data as CSV",
        data=csv,
        file_name=title+".csv",
        mime='text/csv',
    )

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

elif system == '시장 금리 스크래핑':
    MarketRateScrapping()
elif system == 'BackTesing':
    st.markdown("https: // kernc.github.io / backtesting.py /")
    # import matplotlib
    #
    # # Use a backend that doesn't display the plot to the user
    # # we want only to display inside the Streamlit page
    # matplotlib.use('Agg')
    #
    # bt = Backtest(GOOG, SmaCross, commission=.002,
    #               exclusive_orders=True)
    # stats = bt.run()
    # st.write(stats)
    # st.write(type(bt))
    # figure = bt.plot()[0][0]
    # st.write(figure)
    # # show the plot in Streamlit
    # st.pyplot(figure)

