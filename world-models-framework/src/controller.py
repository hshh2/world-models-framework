import tensorflow as tf
from tensorflow.keras import layers, models

def build_controller(state_dim, action_dim):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(state_dim,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(action_dim, activation='softmax'))  # For discrete actions
    return model

if __name__ == "__main__":
    state_dim = 512  # Dimension of the state (output of Vision Module)
    action_dim = 4  # Example action space dimension
    controller = build_controller(state_dim, action_dim)
    controller.summary()