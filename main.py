# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import pandas as pd

from rental_investment_calculator import RentalInvestmentCalculator #임대 수익 계산기
from market_rate import MarketRateScrapping #스크랩핑
from control_excel import convert_df
from macd_straregy import MACDStrategy
from backtesting_strategy import Backtesting
from backtrader_strategy import Backtrader
from stock_data_analysis import CompareStockAnalysis,MDDAnalysis,RelationAnalysis,MonteCarloSimulation
from stock_data_analysis import SharpRatioSimulation,BollingerBandAnalysis,TradingforaLiving,DualMomentumAnalysis

with st.form("시스템 선택"):
    st.header("Python projects of 502")

    system=st.radio("Choice a project", ('MACD Oscillator / RSI 전략 by 502',
                                         '파이썬 증권데이터 분석',
                                         'Backtrader 라이브러리 예제',
                                         'BackTesing 라이브러리 예제',
                                         '시장 금리 스크래핑',
                                         '임대 수익률 계산기',))

    submitted = st.form_submit_button("선택")

if system == 'MACD Oscillator / RSI 전략 by 502':
    MACDStrategy()

elif system == '파이썬 증권데이터 분석':

    selected = st.selectbox('Choose your analysis method.',
                                ('주식 비교하기',
                                 'MDD 구하기',
                                 '주식간 상관관계분석',
                                 'Trading전략 - 몬테카를로 시뮬레이션',
                                 'Trading전략 - 샤프지수 시뮬레이션',
                                 'Trading전략 - 볼랜져밴드 지표분석',
                                 'Trading전략 - 심리투자 법칙',
                                 'Trading전략 - 듀얼모멘텀 투자',
                                 )
                            )

    if selected in '주식 비교하기':
        CompareStockAnalysis()
    elif selected in 'MDD 구하기':
        MDDAnalysis()
    elif selected in '주식간 상관관계분석':
        RelationAnalysis()
    elif selected in 'Trading전략 - 몬테카를로 시뮬레이션':
        MonteCarloSimulation()
    elif selected in 'Trading전략 - 샤프지수 시뮬레이션':
        SharpRatioSimulation()
    elif selected in 'Trading전략 - 볼랜져밴드 지표분석':
        BollingerBandAnalysis()
    elif selected in 'Trading전략 - 심리투자 법칙':
        TradingforaLiving()
    elif selected in 'Trading전략 - 듀얼모멘텀 투자':
        DualMomentumAnalysis()

elif system == 'Backtrader 라이브러리 예제':

    st.markdown("https://www.backtrader.com/")
    Backtrader()

elif system == 'BackTesing 라이브러리 예제':

    st.markdown("https://kernc.github.io/backtesting.py/")
    Backtesting()

elif system == '시장 금리 스크래핑':

    MarketRateScrapping()

elif system == '임대 수익률 계산기':
    st.caption("Sidebar MENU 에서 관련 정보를 입력하세요.")
    result = RentalInvestmentCalculator()

    title = st.text_input("다운받을 파일명을 입력하세요.", max_chars=10)
    add_title = result.rename(columns={0: title})  # 제목 index으로 변경
    csv = convert_df(add_title)

    st.download_button(  # 파일에 쓰고 다운로드
        label="Download data as CSV",
        data=csv,
        file_name=title + ".csv",
        mime='text/csv',
    )

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)


