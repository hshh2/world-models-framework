import tensorflow as tf
from tensorflow.keras import layers, models

def build_vision_module(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    return model

if __name__ == "__main__":
    input_shape = (84, 84, 1)  # Example input shape for grayscale images
    vision_module = build_vision_module(input_shape)
    vision_module.summary()