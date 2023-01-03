import streamlit as st
import pandas as pd
import os
from requests import get  # to make GET request

def WriteExcel(Data):

    file_name = 'test.xlsx'
    df_result = Data

    try:
        writer = pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay')

        max_row = writer.sheets['Sheet1'].max_row

        if max_row == 1:
            df_result.to_excel(
                writer,
                sheet_name='Sheet1',
                startcol = 0,
                startrow = 0,
                index=True,
                encoding = 'utf-8',
                na_rep = '',      # 결측값을 ''으로 채우기
                inf_rep = '',     # 무한값을 ''으로 채우기
                # header = None
                )
            writer.save()
            writer.close()
        else:
            df_result.to_excel(
                writer,
                sheet_name='Sheet1',
                startcol = 0,
                startrow = writer.sheets['Sheet1'].max_row,
                index=True,
                encoding = 'utf-8',
                na_rep = '',      # 결측값을 ''으로 채우기
                inf_rep = '',     # 무한값을 ''으로 채우기
                # header = None
                )
            writer.save()
            writer.close()
    except:
        df_result.to_excel(
            excel_writer = file_name,
            sheet_name = 'Sheet1',
            index = True,
            encoding = 'utf-8',
            na_rep = '',      # 결측값을 ''으로 채우기
            inf_rep = ''     # 무한값을 ''으로 채우기
        )     # 해당 파일이 열려있으면 안됨.

    return "success"

# def DownLoad(file_name):
#     with open(file_name, "wb") as file:   # open in binary mode
#         response = get("http://localhost:8501/test.xlsx")               # get request
#         file.write(response.content)      # write to file


