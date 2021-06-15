#Подключаем библиотеки
import tensorflow as tf
from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set
  _color('red')
  thisplot[true_label].set_color('blue')


# Загружаем MNIST-датасет
fashion_mnist = keras.datasets.fashion_mnist

#  .load_data() -> четыре массива NumPy:
#  1. Массивы train_images и train_labels — это данные, которые использует модель для обучения
#  2. Массивы test_images и test_labels используются для тестирования модели
(train_images,
 train_labels), \
(test_images,
 test_labels) = fashion_mnist.load_data()


# Имена классов одежды
# Индекс в списке соответствует индексу класса
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#В обучающем наборе имеется 60 000 изображений, каждое изображение представлено как 28 x 28 пикселей
print('train images : shape :', train_images.shape)
# В тестовом наборе имеется 10 000 изображений, каждое изображение представлено как 28 x 28 пикселей
print('test images : shape :', test_images.shape)
#В учебном наборе 60 000 меток
print('train labels : len :', len(train_labels))
# В тестовом наборе 10 000 меток
print('train labels : len :', len(test_labels))
# Каждая метка представляет собой целое число от 0 до 9 (Показывается первые 3 метки и последние 3 метки
print('train labels :',train_labels)


# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# Масштабирование значений картинок из [0...255] -> [0...1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Нарисуем первые 25 экзмемпляров тренировочных данных
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# Определение модели
# 1) Первый слой в сети tf.keras.layers.Flatten преобразует формат изображений из 2d-массива (28 на 28 пикселей)
# в 1d-массив из 28 * 28 = 784 пикселей.
# 2) tf.keras.layers.Dense - Первый слой Dense содержит 128 узлов (или нейронов).
# Второй (и последний) уровень — это слой с 10 узлами tf.nn.softmax, который возвращает
# массив из десяти вероятностных оценок, сумма которых равна 1
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Запуск обучения
model.fit(train_images, train_labels, epochs=5)

# Сохранение модели
now = datetime.now()
filepath = 'saved_models/mnist_%i_%i_%i_%i_%i_%i' % (
    now.year, now.month, now.day,
    now.hour, now.minute, now.second
)
print('Saving model...')
print('Path:', filepath)
model.save(filepath=filepath)

# Загрузка модели
print('Loading model: %s...' % filepath)
loaded_model = tf.keras.models.load_model(filepath)

# Тестирование
test_loss, test_acc = loaded_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
print('Test Loss:', test_loss)

# Прогнозирование
predictions = loaded_model.predict(test_images)
print(predictions[0])
print(test_labels[0])

# Визуализация прогнозирования

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

plt.show()

# Прогнозирование для 1 изображения

# Возьмём изображение из тестового набора данных
img = test_images[0]

#Добавим изображение в пакет, где он является единственным членом
img = (np.expand_dims (img, 0))

# Прогноз для изображения:
predictions_single = loaded_model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))


