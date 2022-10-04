from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout


def create_nn(input_size: int = 815, neurons_per_layer=None) -> Model:
    if neurons_per_layer is None:
        neurons_per_layer = list([8, 4, 1])
    input_layer = Input((input_size,))
    x = input_layer
    for neurons in neurons_per_layer[:-1]:
        x = Dense(neurons, activation='relu')(x)
        x = Dropout(0.2)(x)
    x = Dense(neurons_per_layer[-1], activation='sigmoid')(x)
    network = Model(input_layer, x)
    network.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
    return network
