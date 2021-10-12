#
# Streamlit의 아파트 시세 예측 APP.
#

import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt

st.title("아파트 시세 예측 App")

# 아파트 데이터를 불러온다.
df = pd.read_excel("apt/all.xlsx")
df = df[ df['해제사유발생일'].isnull() ]
df.drop(['해제사유발생일'],axis=1,inplace=True)
test = df['시군구'].str.split(" ",expand=True) # 시군구 분할
df['구'] = test.iloc[:,1] # '구' 열 추가
df['동'] = test.iloc[:,2] # '동' 열 추가
df.drop(["시군구"],axis=1,inplace=True) # 시군구 삭제
df['거래금액(만원)'] = df['거래금액(만원)'].str.replace(pat=r'[^\w]', repl=r'', regex=True).astype("float")
df['면적(㎡)당 가격'] = round(df['거래금액(만원)'] / df['전용면적(㎡)']).astype("int")
df['거래금액'] = np.log1p(df['거래금액(만원)'])

# X 데이터를 불러온다.
X = df.drop(['거래금액(만원)','거래금액','면적(㎡)당 가격','계약일','번지','본번','부번','단지명','도로명'],axis=1)
x = pd.get_dummies(X)
y = df['거래금액']

# Y의 유형을 list로 저장해 둔다.
y_labels = ["price"]


# Feature 이름.
my_features_X = ["전용면적(㎡)", "계약년월", "층", "건축년도", "구", "동"]

my_df_X = pd.DataFrame(data=X, columns=my_features_X)

st.sidebar.header("입력해 주세요.")
my_parameters={}
for a_feature in x.columns[0:4]: 
    if a_feature in ["전용면적(㎡)", "계약년월", "층", "건축년도"]:
        a_min = int(x[a_feature].min())
        a_max = int(x[a_feature].max())
        a_mean = int(x[a_feature].mean())
    else:
        a_min = float(x[a_feature].min())
        a_max = float(x[a_feature].max())
        a_mean = float(x[a_feature].mean())
    
    my_parameters[a_feature] =  np.round(st.sidebar.slider(a_feature, a_min, a_max, a_mean),2)

# for a_feature in x.columns[5:6]:
#     my_parameters[a_feature] = x["구"]
#     my_parameters[a_feature] = x["동"]

# my_select_gu = st.sidebar.selectbox("origin_gu", ['구_광산구', '구_남구', '구_동구', '구_북구', '구_서구'])
# 구_광산구, 구_남구, 구_동구, 구_북구, 구_서구 = 0, 0, 0, 0, 0
# if my_select_gu == "광산구":
#     my_parameters["구"] = "광산구"
# elif my_select_gu == "남구":
#     my_parameters["구"] = "남구"
# elif my_select_gu == "동구":
#     my_parameters["구"] = "동구"
# elif my_select_gu == "북구":
#     my_parameters["구"] = "북구"
# else :
#     my_parameters["구"] = "서구"

# my_select_dong = st.sidebar.selectbox("origin_dong", ['각화동', '계림동', '광천동', '금남로2가', '금남로3가', '금남로5가', '금호동',
#        '내방동', '노대동', '농성동', '대인동', '도산동', '도천동', '동림동', '동명동',
#        '동천동', '두암동', '마륵동', '매곡동', '매월동', '문흥동', '방림동', '백운동',
#        '본촌동', '봉선동', '북동', '비아동', '사동', '산수동', '산월동', '산정동',
#        '삼각동', '서동', '선암동', '소촌동', '소태동', '송정동', '송하동', '수기동',
#        '수완동', '신가동', '신안동', '신용동', '신창동', '신촌동', '쌍암동', '쌍촌동',
#        '양동', '양림동', '양산동', '연제동', '오치동', '용두동', '용봉동', '용산동',
#        '우산동', '운남동', '운림동', '운수동', '운암동', '월계동', '월곡동', '월남동',
#        '월산동', '유촌동', '일곡동', '임동', '임암동', '장덕동', '주월동', '중흥동',
#        '지산동', '지석동', '진월동', '충장로4가', '치평동', '풍암동', '풍향동',
#        '하남동', '학동', '행암동', '화정동', '흑석동'])
# # my_parameters["동"] = "각화동"
# if my_select_dong == "각화동":
#     my_parameters["동"] = "각화동"
# 입력된 X 데이터.
st.header("입력된 X 데이터:")
# my_X_raw = np.array([["전용면적(㎡)", "계약년월", "층", "건축년도", "구", "동"]])
my_df_X_raw = pd.DataFrame(data=my_parameters, index=[0])
my_features_X = my_df_X_raw.columns
st.write(my_df_X_raw)


# 전처리된 X 데이터.
with open("my_scaler.pkl","rb") as f:
    my_scaler = pickle.load(f)
# my_X_scaled = my_scaler.transform(my_df_X_raw.values)      # fit_transform이 아닌 transform!!

st.header("전처리된 X 데이터:")
my_df_X_scaled = pd.DataFrame(data=x[:1])
st.write(my_df_X_scaled)

# 예측.
with open("my_regressor.pkl","rb") as f:
    my_regressor = pickle.load(f)

my_Y_pred = my_regressor.predict(my_df_X_scaled)[0]

st.header("예측 결과:")
st.write("예측값: ", np.round(my_Y_pred,2))