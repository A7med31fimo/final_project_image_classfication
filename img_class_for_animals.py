import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
import os
import glob as gb
import cv2

trainpath = './Animals-10/'
testpath = './test/'


s = 60
code = {'butterfly': 0, 'cat': 1, 'chicken': 2, 'cow': 3, 'dog': 4}


def getcode(n):
    for x, y in code.items():
        if n == y:
            return x


X_train = []
y_train = []
for folder in os.listdir(trainpath):
    files = gb.glob(pathname=str(trainpath + '//' + folder + '/*.jpeg'))
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (s, s))
        X_train.append(list(image_array))
        y_train.append(code[folder])
# print(len(X_train))
# print(len(y_train))

X_test = []
y_test = []
for folder in os.listdir(testpath):
    files = gb.glob(pathname=str(testpath + '//' + folder + '/*.jpeg'))
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (s, s))
        X_test.append(list(image_array))
        y_test.append(code[folder])
# print(len(X_test))
# print(len(y_test))


X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)


# print(f'X_train shape  is {X_train.shape}')
# print(f'X_test shape  is {X_test.shape}')
#
# print(f'y_train shape  is {y_train.shape}')
# print(f'y_test shape  is {y_test.shape}')

# scalling
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# catagory the y like if my classes(cow , cat ,dog ) if y=cow so return (1,0,0) if dog return (0,0,1)
# to improve process of learning
y_train_categorical = tf.keras.utils.to_categorical(
    y_train, num_classes=5, dtype='float32'
)
y_test_categorical = tf.keras.utils.to_categorical(
    y_test, num_classes=5, dtype='float32'
)


KerasModel = tf.keras.models.Sequential([
# first layer for convert the 2d dim to 1_d dim
    tf.keras.layers.Flatten(input_shape=(s, s, 3)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])


# use SGD is make model slow in learning but accuracy is better than adam
KerasModel.compile(optimizer='SGD',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


epochs = 60
ThisModel = KerasModel.fit(X_train_scaled, y_train_categorical, epochs=epochs)
KerasModel.save('my_model.h5')
ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test_scaled, y_test_categorical)


print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy))
