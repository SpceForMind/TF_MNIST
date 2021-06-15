from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime

import matplotlib.pyplot as plt




(train_images,
 train_labels), \
(test_images,
 test_labels) = fashion_mnist.load_data()

# Масштабирование значений картинок из [0...255] -> [0...1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Имена классов одежды
# Индекс в списке соответствует индексу класса
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# define number of images to show
num_row = 2
num_col = 8
num= num_row*num_col



# Аугментация с поворотом на 30 градусов
data_generator = ImageDataGenerator(rotation_range=30)
data_generator.fit(train_images.reshape(train_images.shape[0], 28, 28, 1))


# Отображаем данные после аугментации
print('AFTER:\n')
fig2, axes2 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))

## `data_generator.flow()` - принимает данные и лейблы,
## генерирует последовательность аугментированных батчей
for X, Y in data_generator.flow(train_images.reshape(train_images.shape[0], 28, 28, 1),
                                train_labels.reshape(train_labels.shape[0], 1),
                                batch_size=num,
                                shuffle=False):
     for i in range(0, num):
          ax = axes2[i//num_col, i%num_col]
          ax.imshow(X[i].reshape(28,28), cmap='gray_r')
          ax.set_title('Label: {}'.format(int(Y[i])))
     break
plt.tight_layout()
plt.show()


# Дообучение модели
MODEL_PATH = 'saved_models/mnist_2021_6_15_18_4_59'
print('Loading model:', MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

model.fit(x=data_generator.flow(train_images.reshape(train_images.shape[0], 28, 28, 1),
                                train_labels.reshape(train_labels.shape[0], 1),
                                batch_size=num,
                                shuffle=False),
          epochs=5,
          verbose=1)
# Сохранение модели
now = datetime.now()
filepath = 'saved_models/mnist_%i_%i_%i_%i_%i_%i' % (
    now.year, now.month, now.day,
    now.hour, now.minute, now.second
)
print('Saving model...')
print('Path:', filepath)
model.save(filepath=filepath)



