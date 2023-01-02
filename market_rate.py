import streamlit as st
#스크랩핑
import requests
import collections
collections.Callable = collections.abc.Callable #module 'collections' has no attribute 'Callable 에러메세지 개선
from bs4 import BeautifulSoup
import pandas as pd
from html_table_parser import parser_functions
import altair as alt
import plotly.express as px
# import plotly

def MarketRateScrapping():

    st.markdown(':chart_with_upwards_trend: COFIX 출처(은행연합회) : https://portal.kfb.or.kr/fingoods/cofix.php')
    st.markdown(':chart_with_downwards_trend: MOR 출처(국민은행) : https://obank.kbstar.com/quics?page=C019205')
    st.markdown(':chart_with_upwards_trend: KOFIA BIS(금융투자협회) : https://www.kofiabond.or.kr/index.html')
    st.caption('(MOR 3개월 CD금리, MOR 6개월이상 시가평가-채권시장평가기준수익률-금융채AAA)')

    st.subheader(":money_with_wings: COFIX 금리")

    url = 'https://portal.kfb.or.kr/fingoods/cofix.php'
    page = requests.get(url)

    soup = BeautifulSoup(page.text, 'html.parser') #html 로 파싱하여 읽어오기

    data = soup.find_all('table', class_='resultList_ty02') #특정 클래스를 지정하여 data 긁기

    #신규 취급액, 잔액, 신잔액 COFIX 차트 만들기
    table = parser_functions.make2d(data[0]) #긁은 data를 dataframe 으로 만들기 전에 table 형태로 변환

    df=pd.DataFrame(table,columns=table[0]) #첫번째 행을 columns으로 지정하여 dataframe 생성
    df2=df.drop(0) # 첫번째 행 제목 줄 삭제
    df3=df2.drop(df2.columns[0],axis=1)#공시일 삭제 (특정 열 삭제)
    df4 = df3.set_index('대상월')#기준 index 설정
    # st.write(df4)
    df_reversed = df4[::-1] #순서 뒤집기 (금리 낮은 순에서 높은 순으로 바꾸기)
    # st.write(df_reversed)

    #plotly line 차트 설정 활용
    fig = px.line(df_reversed,title='신규 취급액, 잔액, 신 잔액 COFIX 금리 변화',labels={'value':'금리','variable':'항목'},height=450)
    st.plotly_chart(fig)

    #단기 COFIX 차트 만들기
    table = parser_functions.make2d(data[1])

    df = pd.DataFrame(table,columns=table[0])
    df2 = df.drop(0) # 첫번째 제목 줄 삭제
    # st.write(df2)
    df3 = df2.drop(df2.columns[0],axis=1)#공시일 삭제
    # st.write(df3)
    df4 = df3.set_index(df3.columns[0])#기준 index 설정
    # st.write(df4)
    df_reversed = df4[::-1] #순서 뒤집기 (금리 낮은 순에서 높은 순으로 바꾸기)
    # st.write(df_reversed)

    #plotly line 차트 설정 활용
    fig = px.line(df_reversed,title='단기 COFIX 금리 변화',labels={'value':'금리','variable':'항목'},height=550)
    st.plotly_chart(fig)


    st.subheader(":money_with_wings: MOR 금리")

    url = 'https://obank.kbstar.com/quics?page=C019205' #KB MOR 금리
    page = requests.get(url)

    soup = BeautifulSoup(page.text, 'html.parser') #html 로 파싱하여 읽어오기

    data = soup.find_all('table', class_='tType01')  # 특정 클래스를 지정하여 data 긁기
    # st.write(data)
    table = parser_functions.make2d(data[0])

    df = pd.DataFrame(table,columns=table[0])
    # st.write(df)
    df2 = df.drop(0) # 첫번째 제목 줄 삭제
    df3 = df2.drop(1) # 첫번째 제목 줄 삭제
    # st.write(df3)
    df4 = df3.drop(df3.columns[4], axis=1)  # 대상대출(열) 삭제
    df5 = df4.drop(df4.columns[3], axis=1)  # 대상대출(열) 삭제
    # st.write(df5)
    df6 = df5.set_index(df5.columns[0])

    st.write(df6)
    df7= pd.DataFrame.transpose(df6) #행,열 바꾸기
    # st.write(df7)

    fig = px.line(df7,title='MOR 금리 변화(전주-금주)',labels={'value':'금리','index':'주차'},height=550)
    st.plotly_chart(fig)

    #엑셀 : 금융채 data 차트 만들기
    df=pd.read_csv("./data/MOR.csv",encoding='cp949')
    df2=df.set_index(df.columns[0])
    st.write(df2)

    fig = px.line(df2,title='금융채(MOR) 금리 변화',labels={'value':'금리','index':'주차','variable':'항목'},height=550,width=900)
    st.plotly_chart(fig)

