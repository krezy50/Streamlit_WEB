import streamlit as st
import pandas as pd

def rate_of_return_by_interest(Loan,Monthly_Rent_Fee,Input_Price): #이율에 따른 수익률 변화 Data만들기

    loan=Loan
    monthly_rent_fee=Monthly_Rent_Fee
    input_price=Input_Price
    loan_interest = 0.01
    df= {'loan_interest':[],'rate_of_return_loan_cost':[]}

    for i in range(1,150):

        loan_interest = loan_interest + 0.05
        # 월 수령액 (월세 - 이자)
        monthly_revenue = int(monthly_rent_fee - loan * loan_interest / 100 / 12)

        # 대출/비용 수익률
        rate_of_return_loan_cost = monthly_revenue * 12 / input_price * 100

        df['loan_interest'].append(round(loan_interest,2))
        df['rate_of_return_loan_cost'].append(round(rate_of_return_loan_cost,2))

    return df

def FindZeroProfit(Data): #DataFrame 생성과 수익률 0 기준 이율 찾기

    data = Data
    data2 = {}
    data2['이율']=data['loan_interest']
    data2['수익률(임대료)']=data['rate_of_return_loan_cost']

    df=pd.DataFrame(
        data2,
        columns=['이율','수익률(임대료)']
    )
    # st.write(df)
    df2=df.set_index('이율') #index 변경

    #값(수익률) 으로 index(이율) 찾기
    index_number = df2.index[df2['수익률(임대료)']==0.0000]

    return {'df':df2,'index_number':index_number}


def RentalInvestmentCalculator():

    st.sidebar.header(":1234: 임대 수익률 계산기")
    buy_price = st.sidebar.number_input("매입 금액(만원,부가세 제외)",step=1,value=35886)
    vat = st.sidebar.number_input("부가세(만원)",step=1,value=1991)
    area_use = st.sidebar.number_input("전용 평수(평)",step=1,value=21)
    area_all = st.sidebar.number_input("공용 평수(평)",step=1,value=44)
    loan = st.sidebar.number_input("대출 금액(만원)",step=1,value=35800)
    loan_interest  = st.sidebar.number_input("이율(%)",step=0.01,value=5.88)
    deposit = st.sidebar.number_input("보증금(만원)",step=1,value=1300)
    monthly_rent_fee = st.sidebar.number_input("임대료(만원)",step=1,value=130)
    buy_brokerage_fee = st.sidebar.number_input("매입 중개료(만원)",step=1,value=int(buy_price*0.009))
    rent_brokerage_fee = st.sidebar.number_input("임대 중개료(만원)", step=1, value=int((deposit+monthly_rent_fee*100) * 0.009))
    interior = st.sidebar.number_input("인테리어(만원)",step=1,value=100)
    register_brokerage_fee = st.sidebar.number_input("기타. 보험, 법무사 비용 등(만원)",step=1,value=30)


    # 매입 평단가
    average_price_by_area_all =int(buy_price / area_all)
    # 임대 평단가
    average_rent_fee_by_area_all = round(monthly_rent_fee / area_all,2)
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

    st.subheader(":page_facing_up: 기본 정보")
    st.write("매입 금액:", format(buy_price,','),"만원,"," 부가세:", format(vat,','),"만원")
    st.write("전용 평수:", format(area_use,','),"평,"," 공용 평수:",  format(area_all,','),"평")
    # 매입 평단가
    st.write("매입 평단가(공용):", format(average_price_by_area_all,','),"만원")

    st.write("대출 금액", format(loan,','),"만원,"," 이율:", format(round(loan_interest,2),','),"%")

    # 월 이자
    st.write("월 이자:", format(loan_interest_price,','),"만원,"," 임대료:", format(monthly_rent_fee,','),"만원(평당:", format(average_rent_fee_by_area_all,','),"만원)" )

    # 월 수령액 (월세 - 이자)
    st.write("월 수령액 ", format(monthly_rent_fee,','),"-", format(loan_interest_price,','),":", format(monthly_revenue,','),"만원")


    st.subheader(":dollar: 투입 비용")

    # 취등록세
    st.write("취득세:",  format(register_fee,','), "만원")
    st.caption("취득세율 4.6%")

    # 중개료
    st.write("중개료:",  format(brokerage_fee,','), "만원")
    st.caption("매입중개료(매매금액*0.9%) + 임대중개료((보증금+100치월세)*0.9%) + 법무사 비용")

    # 투자금
    st.write("실투자금:", format(input_price,','),"만원")
    st.caption("(매입 금액 - 대출 금액 - 보증금 + 취등록세 + 중개료 + 인테리어)")

    # 초기 투자금
    st.write("초기 투자비:", format(beginning_price,','),"만원")
    st.caption("(실투자금 + 부가세)")

    st.subheader(":moneybag: 수익률")
    # 무대출 수익률
    st.write("자기 자본 수익율(무대출):", format(rate_of_return_no_loan,','),"%")

    # 대출시 수익률
    st.write("대출시 수익률:", format(round(rate_of_return_loan,2),','),"%")
    st.caption("월세x12 / (매입-대출)")

    # 대출/비용 수익률
    st.write("대출/비용 포함시 수익률:", format(round(rate_of_return_loan_cost,2),','),"%")
    st.caption("월세x12 / 실투자금")


    # st.write("이율에 따른 수익률 변화")
    st.markdown(":chart_with_downwards_trend: **:blue[이율에 따른 수익률 변화]**")

    data1 = FindZeroProfit(rate_of_return_by_interest(loan, monthly_rent_fee, input_price))
    data2 = FindZeroProfit(rate_of_return_by_interest(loan, monthly_rent_fee + 10, input_price))
    data3 = FindZeroProfit(rate_of_return_by_interest(loan, monthly_rent_fee - 10, input_price))
    # 라인 차트
    st.line_chart(data1['df'])

    # st.write(data1['index_number'])

    # #값(수익률) 으로 index(이율) 찾기
    # st.write("수익률(입력 임대료) 0%일 때 대출 이율 : ", data1['index_number'][0])
    # st.write("수익률(입력 임대료+10만원) 0%일 때 대출 이율:", data2['index_number'][0])
    # st.write("수익률(입력 임대료-10만원) 0%일 때 대출 이율:", data3['index_number'][0])

    save_data = {

            "매입 금액(만원,부가세 제외)":buy_price,
            "부가세(만원)":vat,
            "전용 평수(평)":area_use,
            "공용 평수(평)":area_all,
            "대출 금액(만원)":loan,
            "이율(%)":loan_interest,
            "보증금(만원)":deposit,
            "임대료(만원)":monthly_rent_fee,
            "매입 중개료(만원)":buy_brokerage_fee,
            "임대 중개료(만원)":rent_brokerage_fee,
            "인테리어(만원)":interior,
            "기타. 보험, 법무사 비용 등(만원)":register_brokerage_fee,
            "매입 평단가(공용):":average_price_by_area_all,
            "월 이자(만원)":loan_interest_price,
            "평당 임대료(만원)":average_rent_fee_by_area_all,
            "월 수령액(만원)":monthly_revenue,
            "취득세(만원)":register_fee,
            "중개료(만원)":brokerage_fee,
            "실투자금(만원)":input_price,
            "초기 투자비(만원)":beginning_price,
            "자기 자본 수익율(무대출)":rate_of_return_no_loan,
            "대출시 수익률":round(rate_of_return_loan,2),
            "대출/비용 포함시 수익률":round(rate_of_return_loan_cost,2),
            # "수익률(입력 임대료) 0%일 때 대출 이율":data1['index_number'][0],
            # "수익률(입력 임대료+10만원) 0%일 때 대출 이율":data2['index_number'][0],
            # "수익률(입력 임대료-10만원) 0%일 때 대출 이율":data3['index_number'][0],

    }

    save_df = pd.DataFrame.from_dict([save_data])
    save_df_transposed = save_df.T
    # st.write(save_df_transposed)

    return save_df_transposed