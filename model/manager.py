from tensorflow import keras
import tensorflow as tf

import os
from datetime import datetime


class ModelManager:
    def __init__(self):
        self.__model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.__loaded = False

    def get_model(self):
        if self.__loaded:
            return self.__model
        else:
            # Компиляция модели
            self.__model.compile(optimizer=tf.keras.optimizers.Adam(),
                                 loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'])
            return self.__model

    def save(self,
             directory_path : str,
             model_name : str):
        now = datetime.now()
        str_now = '%i_%i_%i_%i_%i_%i' % (
            now.year, now.month, now.day,
            now.hour, now.minute, now.second
        )
        path = os.path.join(directory_path, '%s_%s' % (model_name, str_now))
        print('Saving model:', path)
        self.__model.save(filepath=path)

    def load(self,
             path: str):
        self.__loaded = True
        self.__model = tf.keras.models.load_model(path)
        print('Model loaded:', path)