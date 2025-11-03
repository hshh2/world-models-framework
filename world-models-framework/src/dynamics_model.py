import tensorflow as tf
from tensorflow.keras import layers, models

def build_dynamics_model(state_dim, action_dim):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(None, state_dim + action_dim)))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(state_dim)))
    model.add(layers.TimeDistributed(layers.Dense(1)))  # Predict reward
    return model

if __name__ == "__main__":
    state_dim = 512  # Dimension of the state (output of Vision Module)
    action_dim = 4  # Example action space dimension
    dynamics_model = build_dynamics_model(state_dim, action_dim)
    dynamics_model.summary()