
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def neural_net():
    model = Sequential()

    model.add(Dense(128, input_shape=(12,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))

    return model
