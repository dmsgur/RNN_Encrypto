import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from anal_data import getInitName,getCandleData,generateData,confirm_data
from rnn_evaluation import convertValue
import os
def getX(coinname,hanname,timesArr,payment):
    res_summary=0
    cur_high,cur_low,cur_start,cur_end=0,0,0,0
    for ix,times in enumerate(timesArr):
        candle_datas = getCandleData(coinname,times=times,payment=payment)
        if candle_datas["status"]=="0000":
            #시계열 데이터 변경
            source_datas = np.array(candle_datas["data"])
            x_data_start=source_datas[timeslot*-1:, 1].astype("float")
            cur_start+=source_datas[-1:, 1].astype("float")
            #print(x_data_start.shape)
            x_data_end = source_datas[timeslot*-1:, 2].astype("float")
            cur_end += source_datas[-1:, 2].astype("float")
            x_data_high = source_datas[timeslot*-1:, 3].astype("float")
            cur_high += source_datas[-1:, 3].astype("float")
            x_data_low = source_datas[timeslot*-1:, 4].astype("float")
            cur_low += source_datas[-1:, 4].astype("float")
            x_dataset = []
            scaler=None
            for ix in range(len(x_data_start)):
                x_dataset.append(sum([x_data_start[ix], x_data_end[ix],\
                      x_data_high[ix], x_data_low[ix]]) / 4)
            # shape (n,60)
            x_dataset = np.array(x_dataset).reshape((len(x_dataset),-1))
            if os.path.exists(r"models\{}_scaler".format(coinname)):
                with open(r"models\{}_scaler".format(coinname),"rb") as fp:
                    scaler = pickle.load(fp)
            if scaler:
                x_dataset = scaler.fit_transform(x_dataset)
                # shape (n,60,1)
                x_dataset = x_dataset.reshape((1, timeslot, -1))
                # print(x_dataset.shape)
                # print(x_dataset[0])
                rmodel = tf.keras.models.load_model(r"models\{}_{}_rnnmodel.keras".format(coinname,times))
                y_pred = rmodel.predict(x_dataset)
                y_value = convertValue(scaler,y_pred)
                res_summary+=y_value[0][0]
                #print(y_value.shape)
                print("========", times,"========")
                print(" 예측결과값: ", round(y_value[0][0],4))
                print("======================")
            else: print("스케일러 존재하지 않습니다. 해당 가상화폐를 훈련후 실행 바랍니다.")
        else:
            print("데이터 수신 불량")
    #루프종료부
    print("xxxxxxxxxxxxxxxxx",coinname,"xxxxxxxxxxxxxxxxxxx")
    # print(cur_high.shape)#(1,)
    print("현재최고가 :",round(cur_high[0]/len(timesArr),4))
    print("현재시작가 :",round(cur_start[0]/len(timesArr),4))
    print("현재종료가 :",round(cur_end[0]/len(timesArr),4))
    print("현재최하가 :",round(cur_low[0]/len(timesArr),4))
    cur_summary = round(((cur_high[0] / len(timesArr)) + \
                         (cur_start[0] / len(timesArr)) + \
                         (cur_end[0] / len(timesArr)) +\
                         (cur_low[0] / len(timesArr)))/4,4)
    print(" :::::: 현재 요약가 :::::: ",cur_summary)
    res_summary=round(res_summary/len(timesArr),4)
    print(" :::::: 최종 예상 요약 결과 :: ",res_summary,"원 예상")
    gap = res_summary-cur_summary
    print("현재가 대비 예상 :",gap," 원 상승 예측" if gap>0 else "원 하락 예측","({:.2f}%)".format((res_summary/cur_summary-1)*100))
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    
def predict_coinprice(coinname,ctime):
    pass
    rmode = tf.keras.models.load_model(r"models\{}_{}_rnnmodel.keras".format(coinname,ctime))
    #x_data =
names=getInitName()
nameArr = [obj["symbol"] for obj in names]
print(",".join(nameArr))
userInput = input("예측할 화폐 목록을 콤마로 구분하여 작성해주세요, 전체선택은 all 을 입력하세요\n")
if userInput=="all":
    names=names
else:
    userInput = userInput.split(",")
    names=[{"symbol":obj["symbol"],"eng":obj["eng"],"kor":obj["kor"]}\
           for obj in names if obj["symbol"] in userInput]
timeArr = ["24h","12h","4h","10m","3m"]
payment="KRW"
timeslot=60
for coinobj in names:
    coinname = coinobj["symbol"]
    hanname =  coinobj["kor"]
    getX(coinname,hanname,timeArr,payment)
