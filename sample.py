import streamlit as st

y=st.text_input("첫번째 숫자입력:",value=502)
i=st.text_input("두번째 숫자입력:",value=50)


st.write("곱하면?",f":green[{format(int(y)*int(i),',')}]")

st.write("나누면?",f":blue[{format(int(y)/int(i),',')}]")
st.write("더하기?",f":orange[{format(int(y)+int(i),',')}]")
st.write("빼면?",f":red[{format(int(y)-int(i),',')}]")