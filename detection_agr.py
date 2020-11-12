from keras.layers import Input, Dense
from keras.models import Sequential, Model
import argparse
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from keras.optimizers import SGD, Adam
from keras.models import load_model
import time
import pandas as pd
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description="Run detection.py.")
    parser.add_argument('--dataset', type=str, default='AIOPS',
                        help='the dataset of time series.')

    # the parameter of autoEncoder
    parser.add_argument('--AE_stop_epochs', type=int, default=500,
                        help='stop auto_encoder after AE_stop_epochs.')
    parser.add_argument('--AE_batch_size', type=int, default=128,
                        help='the batch_size for training AE.')
    parser.add_argument('--AE_hidden_size', type=list, default=[8, 6, 2],
                        help='The hidden size of auto_encoder.')
    parser.add_argument('--AE_lr', type=float, default=0.001,
                        help='The lr of auto_encoder.')
    parser.add_argument('--AE_beta1', type=float, default=0.9,
                        help='The beta1 of Adam optimizers for AE.')
    parser.add_argument('--AE_beta2', type=float, default=0.999,
                        help='The beta2 of Adam optimizers for AE.')
    parser.add_argument('--AE_epsilon', type=float, default=1e-08,
                        help='The epsilon of Adam optimizers for AE.')

    # the parameter of k-means
    parser.add_argument('--min_cluster', type=int, default=15,
                        help='The min Number of clusters for kmeans.')
    parser.add_argument('--max_cluster', type=int, default=20,
                        help='The most Number of clusters for kmeans.')

    # the parameter of Generate Adversarial Network
    parser.add_argument('--k', type=int, default=5,
                        help='Number of sub_generator.')
    parser.add_argument('--gen_stop_epochs', type=int, default=100,
                        help='Stop training generator after gen_stop_epochs.')
    parser.add_argument('--dis_stop_epochs', type=int, default=125,
                        help='Stop training discriminator after dis_stop_epochs.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--lr_g', type=float, default=0.001,
                        help='Learning rate of generator.')
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='Decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')

    return parser.parse_args()


def create_autoencoder(latent_size, args):
    hidden_size = args.AE_hidden_size
    # encode layer
    en = Sequential()
    en.add(Dense(hidden_size[0], activation='relu', input_dim=latent_size))
    en.add(Dense(hidden_size[1], activation='relu'))
    en.add(Dense(hidden_size[2], activation='relu'))
    en_input = Input(shape=(latent_size,))
    en_out = en(en_input)

    # decode layer
    de = Sequential()
    de.add(Dense(hidden_size[1], activation='relu', input_dim=hidden_size[2]))
    de.add(Dense(hidden_size[0], activation='relu'))
    de.add(Dense(latent_size, activation='relu'))
    de_out = de(en_out)

    # encoder
    encoder = Model(en_input, en_out)

    # autoencoder
    auto_encoder = Model(en_input, de_out)
    return encoder, auto_encoder


def genDataMulti(data_name, cutrate):
    data = pd.read_csv(data_name)
    trainSize = int(data.shape[0] * cutrate)
    testSize = data.shape[0] - trainSize
    X_train = data.iloc[:, :-1].values[:trainSize]
    y_train = data.iloc[:, -1].values[:trainSize]
    X_test = data.iloc[:, :-1].values[-testSize:]
    y_test = data.iloc[:, -1].values[-testSize:]
    print("the label 1 of train %s, and test %s" % (np.sum(y_train), np.sum(y_test)))
    return X_train, y_train, X_test, y_test


def normalLizeData(data, isReshape, scaler):
    if isReshape:
        data = np.reshape(data, (data.shape[0], 1))
        data = scaler.fit_transform(data)
        data = data.flatten()
    else:
        data = scaler.fit_transform(data)
    return data


#聚类算法用于去除噪声
def Denstream(input, args):
    pass


if __name__ == '__main__':
    args = parse_args()
    dataset = 'SMD'
    is_train = False
    start_time = datetime.now()
    strTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    path = '%s.csv' % dataset
    X_train, y_train, X_test, y_test = genDataMulti(path, 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = normalLizeData(X_train, True, scaler)
    data_size = data_norm.shape[0]
    latent_size = data_norm.shape[1]

    encoder, autoencoder = create_autoencoder(latent_size, args)
    autoencoder.compile(optimizer=Adam(lr=args.AE_lr, beta_1=args.AE_beta1, beta_2=args.AE_beta2, epsilon=args.AE_epsilon), loss='mse')

    # train autoencoder
    autoencoder.fit(data_norm, data_norm, epochs=args.AE_stop_epochs, batch_size=args.AE_batch_size)
    auto_represent = encoder.predict(data_norm)
    result = autoencoder.predict(data_norm)