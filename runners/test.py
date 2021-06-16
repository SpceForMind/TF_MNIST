from tensorflow import keras

from model.manager import ModelManager


class TestRunner:
    def __init__(self):
        self.__model_manager = ModelManager()
        self.__dataset = keras.datasets.fashion_mnist

    def run_test(self,
                 model_path: str):
        #  .load_data() -> четыре массива NumPy:
        #  1. Массивы train_images и train_labels — это данные, которые использует модель для обучения
        #  2. Массивы test_images и test_labels используются для тестирования модели
        (train_images,
         train_labels), \
        (test_images,
         test_labels) = self.__dataset.load_data()

        # Тестирование
        self.__model_manager.load(path=model_path)
        test_loss, test_acc = self.__model_manager.get_model().evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)
        print('Test Loss:', test_loss)


def main(args):
    model_path = args.model_path

    test_runner = TestRunner()
    test_runner.run_test(model_path=model_path)