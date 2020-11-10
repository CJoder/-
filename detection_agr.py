from keras import Input, Dense
from keras import Model, Sequential
import numpy as np


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