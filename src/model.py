import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model # type: ignore
from config import IMG_SIZE, NUM_CLASSES


def build_model():
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    for layer in base_model.layers[:-15]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    return Model(inputs=base_model.input, outputs=outputs)
