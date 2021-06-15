import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


MODEL_PATH = 'saved_models/mnist_2021_6_15_18_4_59'
print('Loading model:', MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

(train_images,
 train_labels), \
(test_images,
 test_labels) = fashion_mnist.load_data()

# Масштабирование значений картинок из [0...255] -> [0...1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Тестирование
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
print('Test Loss:', test_loss)
