import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777  # You may select anything up to 60,000
print(y_train[image_index])  # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)
model.evaluate(x_test, y_test)
#
# for i in range(x_train.shape[0]):
#     x = x_train[i].reshape(1, 28, 28, 1)
#     y = y_train[i].reshape(1, 1)
#     model.fit(x, y, verbose=0)
#     if i < x_test.shape[0]:
#         y_predicted = model.predict(x_test[i].reshape(1, 28, 28, 1))
#         y_actual = y_test[i]
#         print("Actual Value: " + str(y_actual) + " & Predicted Value = " + str(np.argmax(y_predicted)))
#     accuracy = model.evaluate(x_test, y_test, verbose=0)
#     print(accuracy)

# image_index = 4444
# plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
# pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print(pred.argmax())
