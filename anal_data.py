import requests
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
def getInitName():#화폐이름 얻어오기
    nameurl = "https://api.bithumb.com/v1/market/all?isDetails=false"
    headers = {"accept": "application/json"}
    response = requests.get(nameurl, headers=headers)
    res_dict = json.loads(response.text)# json.dumps
    # print(res_dict[0].keys())#영문이름
    # print(res_dict[0]["market"])#마켓이름 KRW-BTC KRW 마켓
    # print(res_dict[0]["korean_name"])#한글이름 비트코인
    # print(res_dict[0]["english_name"])#영문이름 Bitcoin
    # print(res_dict[5]["market"])#마켓이름 KRW-QTUM
    # print(res_dict[5]["korean_name"])#한글이름 퀀텀
    # print(res_dict[5]["english_name"])#영문이름 Qtum
    # print(res_dict[100]["market"])#마켓이름 KRW-BFC
    # print(res_dict[100]["korean_name"])#한글이름 바이프로스트
    # print(res_dict[100]["english_name"])#영문이름 Bifrost
    encrypto_names =[]
    for data in res_dict:
        if "KRW-" not in data["market"]:
            print(data["market"])#BTC-ETH  BTC 마켓
            continue
        encrypto_names.append({"symbol":data["market"].split("-")[1],
                               "eng":data["english_name"],
                               "kor":data["korean_name"]})
    return encrypto_names
# 1. 데이터유형 - [{symbol:BTC,eng:bitcoin,kor:비트코인}, ... ]
# print(encrypto_names)
#order_currency 화폐명 BTC
#payment_currency 지불화폐 KRW | BTC
#chart_intervals 데이터간격 차트 간격, 기본값 : 24h {1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h, 6h, 12h, 24h, 1w, 1mm 사용 가능}
def getCandleData(currency="BTC",times="24h",pyament="KRW"):#캔들데이터 얻기
    order_currency = "BTC"
    payment_currency = "KRW"
    chart_intervals = "12h"
    candle_url = f"https://api.bithumb.com/public/candlestick/{order_currency}_{payment_currency}/{chart_intervals}"
    print(candle_url)
    headers = {"accept": "application/json"}
    response = requests.get(candle_url, headers=headers)
    candle_data = json.loads(response.text)
    if candle_data["status"]=='0000':
    #>> [1388070000000*, '737000', '755000', '755000', '737000', '3.78*']
    #>> [기준 시간     ,  시작가  ,  종료가  ,  최고가 ,  최저가 ,   거래량
    #기준시간 변경
        return candle_data
    else :
        return False
    
#정렬순서가 최근 데이터가 맨뒤에
def generateData(source_data,timeslot):#시계열 훈련 데이터 생성
    x_data = [];y_data=[]
    for ix in range(len(source_data)-(timeslot)):
        slot_data =[]
        for cur_ix in range(ix,timeslot+ix):
            slot_data.append(source_data[cur_ix])
            if ix==0 or ix==1:print(source_data[cur_ix])
        x_data.append(slot_data)
        y_data.append(source_data[timeslot+ix])
    return np.array(x_data),np.array(y_data)

def confirm_data(x_data,y_data,source_data):# 문제 데이터와 정답 데이터 일치성 확인
    result_bool=True
    if y_data[0] != x_data[1][-1]:#정답 파일 확인
        result_bool=False
    if y_data[1]!=x_data[2][-1]:
        result_bool=False
    if y_data[-1] != source_data[-1]:#마지막 데이터 확인
        result_bool=False
    if y_data[-2] != source_data[-2]:
        result_bool=False
    return result_bool # True 일때 일치

def scatterAnal(x_data,y_data):#산점도 분석
    #가중평균 적용
    cvdata = np.average(x_data,axis=1,weights=(0.02,0.05,0.05,0.08,0.08,0.1,0.13,0.15,0.16,0.18))
    plt.scatter(cvdata,y_data)
    plt.show()

if __name__=="__main__":
    pass












