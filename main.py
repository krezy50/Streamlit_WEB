# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import pandas as pd

from rental_investment_calculator import RentalInvestmentCalculator #임대 수익 계산기
from market_rate import MarketRateScrapping #스크랩핑
from control_excel import convert_df
from backtesting_straregy import Backtesting
from stock_data_analysis import CompareStockAnalysis,MDDAnalysis,RelationAnalysis


with st.form("시스템 선택"):
    st.header("Python projects of 502")

    system=st.radio("Choice a project", ('임대 수익률 계산기', '시장 금리 스크래핑','BackTesing 예제','파이썬 증권데이터 분석'))
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

elif system == 'BackTesing 예제':

    st.markdown("https://kernc.github.io/backtesting.py/")
    Backtesting()

# elif system == 'Backtrader':
#
#     st.markdown("https://www.backtrader.com/")
#     Backtrader()

elif system == '파이썬 증권데이터 분석':

    CompareStockAnalysis()
    MDDAnalysis()
    RelationAnalysis()
