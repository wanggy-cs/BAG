
import numpy as np
from BHT_ARIMA import BHTARIMA
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense,GRU
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

if __name__ == "__main__":
    pred = []
    data_ts = np.load('input/data.npy').T
    data_s = np.load('input/data.npy')
    print("shape of data: {}".format(data_ts.shape))
    print("This dataset have {} series, and each serie have {} time step".format(
        data_ts.shape[0], data_ts.shape[1]
    ))

    for i in range(data_ts.shape[1]-9):
        ts = data_ts[...,:10+i]
        p = 1
        d = 1
        q = 1
        taus = [18, 2]
        Rs = [5, 5]
        k =  10
        tol = 0.001
        Us_mode = 4

        model = BHTARIMA(ts, p, d, q, taus, Rs, k, tol, verbose=0, Us_mode=Us_mode)
        result, _ = model.run()

        pre = result[..., -1]
        print(pre)
        pred.append(pre)

    np.save('output/pred.npy',pred)
    pred = np.load('output/pred.npy')
    result = np.insert(pred, 0, data_s[:10, ...], axis=0)
    pre = result[-1]
    result_new = np.delete(result, np.s_[364], axis=0)
    residual = data_s - result_new

    training_set = residual[0:364-90, :]
    test_set = residual[364-90:, :]

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    test_set = sc.transform(test_set)


    def create_dataset(data, n_predictions, n_next):
        dim = data.shape[1]
        train_X, train_Y = [], []
        for i in range(data.shape[0] - n_predictions - n_next - 1):
            a = data[i:(i + n_predictions), :]
            train_X.append(a)
            tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
            b = []
            for j in range(len(tempb)):
                for k in range(dim):
                    b.append(tempb[j, k])
            train_Y.append(b)
        train_X = np.array(train_X, dtype='float64')
        train_Y = np.array(train_Y, dtype='float64')
        return train_X, train_Y


    def trainModel(train_X, train_Y):
        model = tf.keras.Sequential([
            GRU(50, return_sequences=True),
            Dropout(0.2),
            GRU(40),
            Dropout(0.2),
            Dense(train_Y.shape[1])
        ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                      loss='mean_squared_error')
        history = model.fit(train_X, train_Y, epochs=30, batch_size=32, verbose=1)
        return model, history



    def reshape_y_hat(y_hat, dim):
        re_y = []
        i = 0
        while i < len(y_hat):
            tmp = []
            for j in range(dim):
                tmp.append(y_hat[i + j])
            i = i + dim
            re_y.append(tmp)
        re_y = np.array(re_y, dtype='float64')
        return re_y


    train_X, train_Y = create_dataset(training_set_scaled, 90, 1)
    np.random.seed(7)
    np.random.shuffle(train_X)
    np.random.seed(7)
    np.random.shuffle(train_Y)
    tf.random.set_seed(7)

    model, history = trainModel(train_X, train_Y)
    loss = history.history['loss']
    plt.plot(loss, label='Training Loss')
    plt.title('Training Loss')
    plt.savefig('output/loss.png')

    test_X = test_set.reshape(1, 90, 18)
    y_pre = model.predict(test_X)
    y_pre = reshape_y_hat(y_pre.flatten(), 18)

    y_pre = sc.inverse_transform(y_pre)
    final_pred = y_pre + pre
    print("prediction of residual error: \n", y_pre)
    print("prediction of BAG: \n", final_pred)











