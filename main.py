# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



import streamlit as st
from rental_investment_calculator import RentalInvestmentCalculator



st.sidebar.header("Python projects of krezy50")
system = st.sidebar.radio("Choice a project",('임대 수익률 계산기','스크래핑'))

if system == '임대 수익률 계산기':
    RentalInvestmentCalculator()


