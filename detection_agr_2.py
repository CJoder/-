import os

from keras.models import Sequential, Model
import argparse
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, MaxPooling1D, RepeatVector, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from datetime import datetime
from keras.optimizers import SGD, Adam
import keras
from keras.models import load_model
import time
import json
import pandas as pd
import math
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description="Run detection.py.")
    parser.add_argument('--dataset', type=str, default='AIOPS',
                        help='the dataset of time series.')

    # the parameter of autoEncoder
    parser.add_argument('--AE_stop_epochs', type=int, default=5,
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


def get_range_proba(predict, label, delay):
    # print(label[1:] != label[:-1])
    # tmp = np.where(label[1:] != label[:-1]) out [tuple,tuple...]
    # 找出变化的地方，也就是后一个数字不同于前面的index，比如label=[0,0,0,1,1,0,1,1,0,1,0]，得到[False False  True False  True  True False  True  True  True]
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0
    # 算法原理，从头开始，先找到第一个预测为1（异常）的点，然后判断pos到延迟delay后是否有标注为1，如果有则该端内都标注为1，否则都标注为0
    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict



def create_generator(latent_size):
    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu',
                  kernel_initializer=keras.initializers.Identity(gain=1.0)))
    gen.add(keras.layers.LeakyReLU(alpha=0.3))
    gen.add(Convolution1D(filters=2, kernel_size=5,strides=1, padding='valid'))
    gen.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    gen.add(keras.layers.LeakyReLU(alpha=0.3))

    gen.add(Convolution1D(filters=2, kernel_size=5,strides=1, padding='valid'))
    gen.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    gen.add(keras.layers.LeakyReLU(alpha=0.3))

    gen.add(Convolution1D(filters=2, kernel_size=5,strides=1, padding='valid'))
    gen.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    gen.add(keras.layers.LeakyReLU(alpha=0.3))
    latent = Input(shape=(latent_size,))
    fake_data = gen(latent)
    return Model(latent, fake_data)


def create_discriminator(data_size, latent_size):
    dis = Sequential()
    dis.add(Dense(math.ceil(math.sqrt(data_size)), input_dim=latent_size, activation='relu',
                  kernel_initializer=keras.initializers.Identity(gain=1.0)))
    dis.add(keras.layers.LeakyReLU(alpha=0.3))
    dis.add(keras.layers.LSTM(latent_size, input_shape=(batch_size, latent_size), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
    dis.add(Convolution1D(filters=2, kernel_size=5,strides=1, padding='valid'))
    dis.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    dis.add(keras.layers.LeakyReLU(alpha=0.3))
    dis.add(Dropout(rate=0.2))

    dis.add(Convolution1D(filters=2, kernel_size=5,strides=1, padding='valid'))
    dis.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    dis.add(keras.layers.LeakyReLU(alpha=0.3))
    dis.add(Dropout(rate=0.2))

    dis.add(Convolution1D(filters=2, kernel_size=5,strides=1, padding='valid'))
    dis.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'))
    dis.add(keras.layers.LeakyReLU(alpha=0.3))
    dis.add(Dropout(rate=0.2))

    dis.add(Dense(latent_size))
    latent = Input(shape=(latent_size,))
    fake_data = dis(latent)
    return Model(latent, fake_data)


def getGuassnoise(batch, latent):
    return np.random.normal(0, 1, (batch, latent))


def save_parameters(path, params):
    fo = open(path, 'w')
    for key in params:
        fo.write(str(key) + '=' + str(params[key]) + '\n')
    fo.close()


#聚类算法用于去除噪声
def Denstream(input, args):
    pass


if __name__ == '__main__':
    args = parse_args()
    dataset = 'SMD'
    is_train = True
    anomaly_rate = 0.05
    epoch = 100
    stop = 0
    generator_loss = 0
    globalAUC = -9999
    start_time = datetime.now()
    train_history = defaultdict(list)
    strTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model2SaveBest = 'run/SMD/model/BestGAN.h5'
    args2Dict = vars(args)

    path = '%s.csv' % dataset
    X_train, y_train, X_test, y_test = genDataMulti(path, 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = normalLizeData(X_train, False, scaler)
    data_size = data_norm.shape[0]
    latent_size = data_norm.shape[1]

    encoder, autoencoder = create_autoencoder(latent_size, args)
    autoencoder.compile(optimizer=Adam(lr=args.AE_lr, beta_1=args.AE_beta1, beta_2=args.AE_beta2, epsilon=args.AE_epsilon), loss='mse')

    # train autoencoder
    autoencoder.fit(data_norm, data_norm, epochs=args.AE_stop_epochs, batch_size=args.AE_batch_size)
    auto_represent = encoder.predict(data_norm)
    result = autoencoder.predict(data_norm)
    np.maximum(auto_represent, 0.0001)
    X = auto_represent + 0.0001
    bandwidth = estimate_bandwidth(X, quantile=0.15, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    clusters_index = defaultdict(list)
    for i in range(X.shape[0]):
        for j in range(n_clusters_):
            if labels[i] == j:
                clusters_index['cluster' + str(j)].append(i)
                break
    sum = 0
    for i in range(n_clusters_):
        len_cluster = len(clusters_index['cluster' + str(i)])
        sum += len_cluster
        print(len_cluster)

    min_len_cluster = X.shape[0] * anomaly_rate
    min_cluster_index = []
    for i in range(n_clusters_):
        len_cluster = len(clusters_index['cluster' + str(i)])
        if len_cluster < min_len_cluster:
            min_cluster_index.append(i)
    num_labeled0 = 0
    data_negtive = []
    for i in min_cluster_index:
        num_labeled0 = num_labeled0 + len(clusters_index['cluster' + str(i)])
        for j in clusters_index['cluster' + str(i)]:
            data_negtive.append(j)

    # 去除异常数据
    # data_positive = np.array([])
    X_train = np.delete(X_train, data_negtive, axis=0)
    y_train = np.delete(y_train, data_negtive, axis=0)
    data_positive = X_train
    data_positive_size = X_train.shape[0]

    #reshape input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    n_timesteps, n_features, n_outputs = X_train[1], X_train[2], 1
    y_train = y_train.reshape((y_train.shape[0], 1, 1))
    print(X_train.shape, y_train.shape)

    lstm_cnn = Sequential()
    lstm_cnn.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(5, 1)))
    lstm_cnn.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    lstm_cnn.add(MaxPooling1D(pool_size=2, padding='same'))
    lstm_cnn.add(Flatten())
    lstm_cnn.add(RepeatVector(1))
    lstm_cnn.add(LSTM(200, activation='relu', return_sequences=True))
    lstm_cnn.add(Dropout(0.1))
    lstm_cnn.add(Dense(1, activation='sigmoid'))
    lstm_cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    lstm_cnn.summary()
    lstm_cnn.fit(X_train, y_train, epochs=100, batch_size=512, verbose=1, validation_split=0.1)

    y_pred = lstm_cnn.predict(X_test)
    AUC = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('AUC:{}, precision:{}, recall:{}, f1:{}'.format(AUC, precision, recall, f1))
    #训练GAN
    # if is_train:
    #     params2Save = 'run/SMD/result/Params.txt'
    #     generator = create_generator(latent_size)
    #     generator.compile(optimizer=SGD(lr=args.lr_g, decay=args.decay, momentum=args.momentum),
    #                       loss='binary_crossentropy')
    #     discriminator = create_discriminator(data_size, latent_size)
    #     discriminator.compile(optimizer=SGD(lr=args.lr_d, decay=args.decay, momentum=args.momentum),
    #                           loss='binary_crossentropy')
    #
    #     for epoch in range(args.dis_stop_epochs):
    #         batch_size = min(512, data_positive_size)
    #         num_batches = int(data_positive_size / batch_size)
    #         for index in range(num_batches):
    #             noise = getGuassnoise(batch_size, latent_size)
    #             data_batch = data_positive[index * batch_size: (index + 1) * batch_size]
    #             generated_data = generator.predict(noise, verbose=0)
    #             X = np.concatenate(data_batch, generated_data)
    #             Y = np.concatenate((np.array([0] * batch_size), np.array([1] * batch_size)))
    #
    #             # Train discriminator
    #             discriminator_loss = discriminator.train_on_batch(X, Y)
    #             print("discriminator_loss: " + str(discriminator_loss))
    #             train_history['discriminator_loss'].append(discriminator_loss)
    #
    #             p_value = discriminator.predict(data_positive)
    #             # Train generator
    #             noise_2 = getGuassnoise(batch_size, latent_size)
    #             if stop == 0:
    #                 generator.train_on_batch(noise_2, data_batch)
    #             else:
    #                 generator_loss = generator.evaluate(noise_2, data_batch, verbose=0)
    #
    #             train_history['generator_loss'].append(generator_loss)
    #             print("generator_loss: " + str(generator_loss))
    #             # Stop training generator
    #             if epoch + 1 > args.gen_stop_epochs:
    #                 stop = 1
    #
    #         # Detection result
    #         y_pred = discriminator.predict(data_norm)
    #
    #         # 计算AUC
    #         AUC = roc_auc_score(y_train, y_pred)
    #
    #         outliers_fraction = np.count_nonzero(y_train) / len(y_train)
    #         threshold = np.percentile(y_pred, 100 * (1 - outliers_fraction))
    #         y_pred = (y_pred > threshold).astype('int')
    #         print(np.sum(y_pred))
    #         y_pred = get_range_proba(y_pred, y_test, 2)
    #         print(np.sum(y_pred))
    #         y_test = np.array(y_test)
    #
    #         precision = precision_score(y_train, y_pred)
    #         recall = recall_score(y_train, y_pred)
    #         f1 = f1_score(y_train, y_pred)
    #
    #         if AUC > globalAUC:
    #             discriminator.save(model2SaveBest)
    #             globalAUC = AUC
    #             args2Dict['AUC'] = AUC
    #             args2Dict['precision'] = precision
    #             args2Dict['recall'] = recall
    #             args2Dict['f1'] = f1
    #
    #         print('AUC:{}, precision:{}, recall:{}, f1:{}'.format(AUC, precision, recall, f1))
    #
    #
    #     # for i in range(num_batches):
    #     #     train_history['auc'].append(AUC)
    #     now_time = datetime.now()
    #     useSecond = (now_time - start_time).seconds
    #     useTime = time.strftime("%H:%M:%S", time.gmtime(useSecond))
    #     print("Have use time: " + useTime + "\n")
    #
    #     args2Dict['usedTime'] = useTime
    #     print('\n' + json.dumps(args2Dict))
    #
    #     save_parameters(params2Save, args2Dict)
    # else:
    #     pass