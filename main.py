# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
from rental_investment_calculator import RentalInvestmentCalculator #임대 수익 계산기
from market_rate import MarketRateScrapping #스크랩핑


st.sidebar.header("Python projects of 502")
system = st.sidebar.radio("Choice a project",('임대 수익률 계산기','시장 금리 스크래핑'))

if system == '임대 수익률 계산기':
    RentalInvestmentCalculator()
elif system == '시장 금리 스크래핑':
    MarketRateScrapping()

