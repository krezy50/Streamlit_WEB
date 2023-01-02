# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
from rental_investment_calculator import RentalInvestmentCalculator #임대 수익 계산기
from market_rate import MarketRateScrapping #스크랩핑


with st.form("시스템 선택"):
    st.header("Python projects of 502")

    system=st.radio("Choice a project", ('임대 수익률 계산기', '시장 금리 스크래핑'))
    submitted = st.form_submit_button("Submit")

if system == '임대 수익률 계산기':
    st.caption("sidebar menu에서 관련 정보를 입력하세요.")
    RentalInvestmentCalculator()
elif system == '시장 금리 스크래핑':
    MarketRateScrapping()

