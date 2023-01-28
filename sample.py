import streamlit as st

y=st.text_input("첫번째 숫자입력:",value=1)
i=st.text_input("두번째 숫자입력:",value=2)


st.write("정답은",int(y)+int(i))