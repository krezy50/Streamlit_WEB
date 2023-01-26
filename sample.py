import streamlit as st

y=st.text_input("첫번째 숫자입력:")
i=st.text_input("두번째 숫자입력:")


st.write(int(y)+int(i))