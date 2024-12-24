import  numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from anal_data import getInitName,getCandleData,generateData,confirm_data,scatterAnal
from utility import cv_format,cv_mill2date,cv_date2milli,cv_str2date,gapCompare

#1.화폐이름 목록 추출
names=getInitName()
print(names[0])
#2. 화폐 캔들 데이터 수신
#getCandleData(currency="BTC",times="24h",pyament="KRW")
candle_datas = getCandleData(names[0]["symbol"],times="24h",pyament="KRW")
print(candle_datas.keys())

if candle_datas["status"]=="0000":
    #3. 훈련 데이터 생성
    # [기준 시간     ,  시작가  ,  종료가  ,  최고가 ,  최저가 ,   거래량]
    source_datas = np.array(candle_datas["data"])
    print()
    x_data_start,y_data_start = generateData(source_datas[:,1],30)#(source_data,timeslot)
    print(x_data_start.shape,y_data_start.shape)
    #4. 데이터 일치성확인
    res = confirm_data(x_data_start,y_data_start,source_datas[:,1])
    if res:
        print("모든데이터 정답과 일치")
    else:print("데이터 혼합 잘못됨")

else:print("데이터 수신 실패")