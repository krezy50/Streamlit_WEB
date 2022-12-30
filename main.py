# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import streamlit as st
import pandas as pd
import altair as alt

def rate_of_return_by_interest(Loan,Monthly_Rent_Fee,Input_Price):

    loan=Loan
    monthly_rent_fee=Monthly_Rent_Fee
    input_price=Input_Price
    loan_interest = 0.01
    df= {'loan_interest':[],'rate_of_return_loan_cost':[]}

    for i in range(1,700):

        loan_interest = loan_interest + 0.01
        # 월 수령액 (월세 - 이자)
        monthly_revenue = int(monthly_rent_fee - loan * loan_interest / 100 / 12)

        # 대출/비용 수익률
        rate_of_return_loan_cost = monthly_revenue * 12 / input_price * 100

        df['loan_interest'].append(round(loan_interest,2))
        df['rate_of_return_loan_cost'].append(round(rate_of_return_loan_cost,2))


    return df


def main():

    buy_price = st.sidebar.number_input("매입 금액(만원,부가세 제외)",step=1,value=35886)
    vat = st.sidebar.number_input("부가세(만원)",step=1,value=1991)
    area_use = st.sidebar.number_input("전용 평수(평)",step=1,value=21)
    area_all = st.sidebar.number_input("공용 평수(평)",step=1,value=44)
    loan = st.sidebar.number_input("대출 금액(만원)",step=1,value=35800)
    loan_interest  = st.sidebar.number_input("이율(%)",step=0.01,value=5.88)
    deposit = st.sidebar.number_input("보증금(만원)",step=1,value=1400)
    monthly_rent_fee = st.sidebar.number_input("월세(만원)",step=1,value=140)
    buy_brokerage_fee = st.sidebar.number_input("매입 중개료(만원)",step=1,value=int(buy_price*0.009))
    rent_brokerage_fee = st.sidebar.number_input("임대 중개료(만원)", step=1, value=int((deposit+monthly_rent_fee*100) * 0.009))
    register_brokerage_fee = st.sidebar.number_input("법무사 비용(만원)",step=1,value=30)
    interior = st.sidebar.number_input("인테리어(만원)",step=1,value=1500)

    # 매입 평단가
    average_price_by_area_all =int(buy_price / area_all)
    # 무대출 수익률
    rate_of_return_no_loan = round(monthly_rent_fee*12/buy_price*100,2)
    # 월 이자
    loan_interest_price = int(loan*loan_interest/100/12)
    # 월 수령액 (월세 - 이자)
    monthly_revenue = int(monthly_rent_fee-loan*loan_interest/100/12)
    # 취등록세
    register_fee = int(buy_price*0.046)
    # 중개료
    brokerage_fee = buy_brokerage_fee + rent_brokerage_fee + register_brokerage_fee
    # 투자금
    input_price = buy_price - loan - deposit + register_fee + brokerage_fee +interior
    # 대출 수익률
    rate_of_return_loan = monthly_revenue*12/(buy_price-loan)*100
    # 대출/비용 수익률
    rate_of_return_loan_cost = monthly_revenue*12/input_price*100
    # 초기 투자금
    beginning_price = input_price + vat

    st.subheader("입력 정보")
    st.write("매입 금액:", buy_price,"만원,"," 부가세:", vat,"만원")
    st.write("전용 평수:", area_use,"평,"," 공용 평수:", area_all,"평")
    st.write("대출 금액",loan,"만원,"," 이율:",round(loan_interest,2),"%")

    st.subheader("분석 결과")
    # 매입 평단가
    st.write("매입 평단가(공용):",average_price_by_area_all,"만원")

    # 무대출 수익률
    st.write("자기 자본 수익율:",rate_of_return_no_loan,"%")

    # 월 이자
    st.write("월 이자:",loan_interest_price,"만원")

    # 월 수령액 (월세 - 이자)
    st.write("월 수령액 ",monthly_rent_fee,"-",loan_interest_price,":",monthly_revenue,"만원")

    st.subheader("추가 비용")

    # 취등록세
    st.write("취득세:",register_fee,"만원")
    st.caption("(4.6%)")
    
    # 중개료
    st.write("중개료:",brokerage_fee,"만원")
    st.caption("매입중개료(매매금액*0.9%) + 임대중개료((보증금+100치월세)*0.9%) + 법무사 비용")

    # 대출 수익률
    st.write("매입-대출 수익률:",round(rate_of_return_loan,2),"%")
    st.caption("월세x12 / (매입-대출)")

    # 대출/비용 수익률
    st.write("대출/비용 포함 수익률:",round(rate_of_return_loan_cost,2),"%")
    st.caption("월세x12 / 실투자금")
    
    # st.write("이율에 따른 수익률 변화")

    data = rate_of_return_by_interest(loan,monthly_rent_fee,input_price)

    df=pd.DataFrame(
        data,
    )
 
    chart = (
        alt.Chart(
            data=df,
            title="이율에 따른 수익률 변화",
        )
        .mark_line()
        .encode(
            x=alt.X("loan_interest", axis=alt.Axis(title="이율")),
            y=alt.Y("rate_of_return_loan_cost", axis=alt.Axis(title="수익률")),
        )
    )

    st.altair_chart(chart, use_container_width=True)


    st.subheader("투입 비용")
    # 투자금
    st.write("실투자금:",input_price,"만원")
    st.caption("(매입 금액 - 대출 금액 - 보증금 + 취등록세 + 중개료 + 인테리어)")

    # 초기 투자금
    st.write("초기 투자비:",beginning_price,"만원")
    st.caption("(실투자금 + 부가세)")



main()