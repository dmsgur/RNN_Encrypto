import matplotlib.pyplot as plt
def rnn_graph(fit_his):
    plt.subplot(1,2,1)
    plt.plot(fit_his.history["acc"],label="train acc")
    plt.plot(fit_his.history["val_acc"],label="valid acc")
    plt.legend()
    plt.title("ACCURACY")
    plt.subplot(1,2,2)
    plt.plot(fit_his.history["loss"],label="train loss")
    plt.plot(fit_his.history["val_loss"],label="valid loss")
    plt.legend()
    plt.title("MAE LOSS")
    plt.show()
def convertValue(scaler,val):
    return scaler.inverse_transform(val)
def today_predict(rmodel,today_x):
    return rmodel.predict(today_x)
def evaluationModel(rmodel,xd,yd):
    return rmodel.evaluate(xd,yd)




