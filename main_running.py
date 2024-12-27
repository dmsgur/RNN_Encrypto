import  numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os
from anal_data import getInitName,getCandleData,generateData,confirm_data,scatterAnal
from utility import cv_format,cv_mill2date,cv_date2milli,cv_str2date,gapCompare
from RNN_constructure import constructureModel
from rnn_evaluation import rnn_graph,convertValue,today_predict,evaluationModel

def train_running(coinname,hanname,timesArr,payment):
    for ix,times in enumerate(timesArr):
        print(f"{len(timesArr)-ix}회 남음: {coinname}({hanname}_{times} 훈련시작)")
        #names[0]["symbol"]
        #2. 화폐 캔들 데이터 수신
        #getCandleData(currency="BTC",times="24h",pyament="KRW")
        candle_datas = getCandleData(coinname,times=times,payment=payment)
        print(candle_datas.keys())
        if candle_datas["status"]=="0000":
            #3. 훈련 데이터 생성
            # [기준 시간     ,  시작가  ,  종료가  ,  최고가 ,  최저가 ,   거래량]
            source_datas = np.array(candle_datas["data"])
            print()
            x_data_start,y_data_start = generateData(source_datas[:,1],timeslot)#(source_data,timeslot)
            print(x_data_start.shape,y_data_start.shape)
            #4. 데이터 일치성확인
            res = confirm_data(x_data_start,y_data_start,source_datas[:,1])
            if res:
                print("모든데이터 정답과 일치")
            else:print("데이터 혼합 잘못됨")
            # scatterAnal(x_data_start, y_data_start,weight_avg,"start price")
            x_data_end, y_data_end = generateData(source_datas[:, 2], timeslot)  # (source_data,timeslot)
            res = confirm_data(x_data_end, y_data_end, source_datas[:, 2])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합 잘못됨")
            # scatterAnal(x_data_end, y_data_end, weight_avg, "end price")
            x_data_high, y_data_high = generateData(source_datas[:, 3], timeslot)  # (source_data,timeslot)
            res = confirm_data(x_data_high, y_data_high, source_datas[:, 3])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합 잘못됨")
            # scatterAnal(x_data_high, y_data_high, weight_avg, "maxhigh price")
            x_data_low, y_data_low = generateData(source_datas[:, 4], timeslot)  # (source_data,timeslot)
            res = confirm_data(x_data_low, y_data_low, source_datas[:, 4])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합 잘못됨")
            # scatterAnal(x_data_low, y_data_low, weight_avg, "minlow price")
            x_data_amount, y_data_amount = generateData(source_datas[:, 5], timeslot)  # (source_data,timeslot)
            res = confirm_data(x_data_amount, y_data_amount, source_datas[:, 5])
            if res:
                print("모든데이터 정답과 일치")
            else:
                print("데이터 혼합 잘못됨")
            # scatterAnal(x_data_amount, y_data_amount, weight_avg, "trade quantity")
            #분석결과 5 인덱스의 quantity 부분은 선형선과 관련부족으로 제외

        else:print("데이터 수신 실패")

        #최종 데이터 생성
        print(x_data_start.shape)
        print(type(x_data_start))
        x_dataset = []
        for ix in range(len(x_data_start)):
            x_d=[]
            for tx in range(timeslot):
                x_d.append(
                    sum([x_data_start[ix][tx],x_data_end[ix][tx],\
                        x_data_high[ix][tx],x_data_low[ix][tx]])/4)
            x_dataset.append(x_d)
        y_dataset=[]
        for ix in range(len(y_data_start)):

            y_dataset.append(
                sum([y_data_start[ix], y_data_end[ix], \
                    y_data_high[ix], y_data_low[ix]]) / 4)
        x_data = np.array(x_dataset)
        y_data = np.array(y_dataset).reshape((len(y_dataset),-1))
        print(x_data.shape)
        print(y_data.shape)
        print(type(x_data[0]))
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        #LSTM 입력차원 dim 3
        x_data = scaler.fit_transform(x_data).reshape((len(x_data),timeslot,-1))
        y_data = scaler.fit_transform(y_data)
        print("dim:",x_data.ndim)
        print(x_data[0][:1])
        print(y_data[:1])
        rmodel=None
        if not os.path.exists(r"models\{}_{}_rnnmodel.keras".format(coinname,times)):
            #LSTM 모델 생성
            rmodel = constructureModel(timeslot)
        else:
            rmodel = tf.keras.models.load_model(r"models\{}_{}_rnnmodel.keras".format(coinname,times))
        print(x_data.shape)
        print(y_data.shape)
        fit_his = rmodel.fit(x_data,y_data,validation_data=(x_data,y_data),epochs=count_epoch,batch_size=len(x_data)//10)
        rmodel.save(r"models\{}_{}_rnnmodel.keras".format(coinname,times))
        with open(r"models\{}_{}_fit_his".format(coinname,times)) as fp:
            pickle.dump(fit_his, fp)
        if not os.path.exists(r"models\{}_scaler".format(coinname)):
            with open(r"models\{}_scaler".format(coinname), "wb") as fp:
                pickle.dump(scaler, fp)
        loss,acc = evaluationModel(rmodel,x_data,y_data)
        print("손실도 : ", loss, " 정확도 :",acc)
        rarr = np.random.randint(0,len(x_data)-2,9)
        test_x = []
        test_y = []
        rarr = np.append(rarr,[len(x_data)-1],axis=0)
        for i in rarr:
            test_x.append(x_data[i])
            test_y.append(y_data[i])
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        pred_y = today_predict(rmodel,test_x)
        true_value = convertValue(scaler,test_y)
        pred_value = convertValue(scaler,pred_y)
        for i in range(len(true_value)):
            print(ix+1,". 실제값:",round(true_value[i][0],4)," 예측값:",round(pred_value[i][0],4))
        rat = (np.abs(pred_value - true_value) / true_value).sum()/len(true_value) * 100
        print(rat)
        print("{}({}) {} 실제값과 예측값 오차율 : {:.2f}".\
              format(coinname,hanname,times,rat,"%"))

#1.화폐이름 목록 추출
names=getInitName()
nameArr = [obj["symbol"] for obj in names]
print(",".join(nameArr))
userInput = input("분석할 화폐 목록을 콤마로 구분하여 작성해주세요, 전체선택은 all 을 입력하세요")
if userInput=="all":
    names=names
else:
    userInput = userInput.split(",")
    names=[{"symbol":obj["symbol"],"eng":obj["eng"],"kor":obj["kor"]}\
           for obj in names if obj["symbol"] in userInput]
timeslot = 60
count_epoch = 3
weight_avg = np.linspace(0,1,timeslot)
if len(weight_avg)!=timeslot:
    print("가중치와 타임슬롯 수량을 동일하게 맞춰주세요")

#24h {1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h, 6h, 12h, 24h, 1w, 1mm 사용 가능}
timeArr = ["24h","12h","4h","10m","3m"]
payment="KRW"
for coinobj in names:
    coinname = coinobj["symbol"]
    hanname =  coinobj["kor"]
    train_running(coinname,hanname,timeArr,payment)
rnn_graph(userInput,timeArr)














