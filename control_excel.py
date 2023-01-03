import streamlit as st
import pandas as pd

def WriteExcel(Data):

    file_name = 'test.xlsx'
    df_result = Data

    try:
        writer = pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay')

        max_row = writer.sheets['Sheet1'].max_row
        # max_col = writer.sheets['Sheet1'].max_col

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
                startrow = max_row,
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
            index = False,
            encoding = 'utf-8',
            na_rep = '',      # 결측값을 ''으로 채우기
            inf_rep = ''     # 무한값을 ''으로 채우기
        )     # 해당 파일이 열려있으면 안됨.

    return "save success"

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


