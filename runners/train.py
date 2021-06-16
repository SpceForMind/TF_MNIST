from tensorflow import keras
import os

from model.manager import ModelManager


class TrainRunner:
    def __init__(self):
        self.__model_manager = ModelManager()
        self.__dataset = keras.datasets.fashion_mnist

    def run_train(self,
                  epochs: int=5,
                  model_path: str=None):
        '''Training Loop on default FASHION MNIST dataset

        :param epochs:
        :param model_path: if param is None -> train model from scratch
        :return:
        '''
        if model_path is not None:
            self.__model_manager.load(path=model_path)

        #  .load_data() -> четыре массива NumPy:
        #  1. Массивы train_images и train_labels — это данные, которые использует модель для обучения
        #  2. Массивы test_images и test_labels используются для тестирования модели
        (train_images,
         train_labels), \
        (test_images,
         test_labels) = self.__dataset.load_data()

        # Обучение
        self.__model_manager.get_model().fit(x=train_images,
                                             y=train_labels,
                                             epochs=epochs)

        # Сохраненние модели
        model_dir = os.path.dirname(model_path) if model_path is not None \
            else 'saved_models'
        model_name = 'fashion_mnist_train_boosted' if model_path is not None \
            else 'fashion_mnist_train'
        self.__model_manager.save(directory_path=model_dir, model_name=model_name)


def main(args):
    model_path = args.model_path
    epochs = args.epochs

    train_runner = TrainRunner()
    train_runner.run_train(epochs=epochs, model_path=model_path)