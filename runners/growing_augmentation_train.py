import os
from tensorflow import keras

from model.manager import ModelManager

from augmentation.manager import AugmentationManager

from keras.callbacks import EarlyStopping

from ml_flow_setup.base import BaseMLFlowSetup


class AugmentationTrainRunner:
    def __init__(self):
        self.__model_manager = ModelManager()
        self.__augmentation_manager = AugmentationManager()

    def run_train(self,
                  epochs: int=10,
                  multiplicity: int=1,
                  model_path: str=None):
        '''Training Loop on default FASHION MNIST dataset

        :param epochs:
        :param model_path: if param is None -> train model from scratch
        :return:
        '''
        if model_path is not None:
            self.__model_manager.load(path=model_path)

        augmented_train_images, ext_train_labels = self.__augmentation_manager.growing_augmentation(
            multiplicity=multiplicity)

        (train_images,
         train_labels), \
        (test_images,
         test_labels) = keras.datasets.fashion_mnist.load_data()

        # Обучение
        history = self.__model_manager.get_model().fit(x=augmented_train_images,
                                                       y=ext_train_labels,
                                                       epochs=epochs,
                                                       validation_data=(test_images, test_labels),
                                                       callbacks=[
                                                           EarlyStopping(
                                                               monitor='val_accuracy',
                                                               patience=5
                                                           )
                                                       ])
        metrics = history.history
        ml_flow_setup = BaseMLFlowSetup()
        ml_flow_setup.log_metrics(experiment_name='fashion_mnist',
                                  args_dict={
                                      'epochs': epochs,
                                      'multiplicity': multiplicity
                                  },
                                  growing_aug_train_loss=metrics['loss'],
                                  growing_aug_train_acc=metrics['accuracy'],
                                  growing_aug_val_loss=metrics['val_loss'],
                                  growing_aug_val_acc=metrics['val_accuracy']
        )

        # Сохраненние модели
        model_dir = os.path.dirname(model_path) if model_path is not None \
            else 'saved_models'
        model_name = 'growing_augmentation_fashion_mnist_train_boosted' if model_path is not None \
            else 'growing_augmentation_fashion_mnist_train'
        self.__model_manager.save(directory_path=model_dir, model_name=model_name)


def main(args):
    model_path = args.model_path
    epochs = args.epochs
    multiplicity = args.multiplicity

    train_runner = AugmentationTrainRunner()
    train_runner.run_train(epochs=epochs, multiplicity=multiplicity, model_path=model_path)


