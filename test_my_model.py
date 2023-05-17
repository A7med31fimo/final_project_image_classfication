import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
import glob as gb
import cv2

KerasModel = tf.keras.models.load_model('./my_model_95%.h5')

testpath = './test/'
predpath = './pred/'

s = 60
code = {'butterfly': 0, 'cat': 1, 'chicken': 2, 'cow': 3, 'dog': 4}


def getcode(n):
    for x, y in code.items():
        if n == y:
            return x



X_pred = []
y_animal = []
for folder in os.listdir(testpath):
    files = gb.glob(pathname=str(testpath + '//' + folder + '/*.jpeg'))
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (s, s))
        X_pred.append(list(image_array))
        y_animal.append(file.split(" ")[0].split("\\")[1])
print(len(X_pred))
print(len(y_animal))

X_pred_array = np.array(X_pred)



y_result = KerasModel.predict(X_pred_array)
count = 0
# plt.figure(figsize=(20, 20))
for i in range(len(y_result)):
    # plt.subplot(6, 6, n + 1)
    # plt.imshow(X_pred[i])
    # plt.axis('off')
    # plt.title()
    # print("Expetected :", getcode(np.argmax(y_result[i])), "The Real :", y_animal[i])
    count += 1 if getcode(np.argmax(y_result[i])) == y_animal[i] else 0
print(count/len(y_result))
# plt.show()
