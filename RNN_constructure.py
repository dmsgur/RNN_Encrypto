import tensorflow as tf
from keras.src.layers import Dropout
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense,LSTM,Dropout
def constructureModel(timeslot):
    rmodel = Sequential()
    rmodel.add(Input(shape=(timeslot,1)))
    # units, activation='tanh', recurrent_activation='sigmoid',dropout=0.0,
    # recurrent_dropout=0.0, return_sequences=False,  return_state=False,
    # return_sequences=True 이면 순회시 가중치를 모두 추출
    # return_sequences=False 이면 마지막 가중치를 모두 추출
    rmodel.add(LSTM(128,activation="relu",dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
    rmodel.add(Dropout(0.2))
    rmodel.add(LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    rmodel.add(Dropout(0.2))
    rmodel.add(LSTM(64, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    rmodel.add(Dropout(0.2))
    rmodel.add(Dense(1))#회귀문제로 가중치를 그대로 출력한다.
    rmodel.compile(loss="mse",optimizer="adam",metrics=["acc"])
    return rmodel



