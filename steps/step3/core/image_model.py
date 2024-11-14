from dataclasses import dataclass
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from typing import Tuple
import numpy as np
import os

@dataclass
class CNNModel:
    _image_size = (1, 32, 32, 3)
    _classes = ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']
    _file_dir = 'storage/'
    file_path = ''
    _model = None

    def __init__(cls):
        files = os.listdir(cls._file_dir)   
        cls.file_path = cls._file_dir+files[-1]
        cls._model = load_model('./core/sesac_cnn2.h5')

    def predict(cls) -> Tuple:
        img = image.load_img(cls.file_path,target_size=cls._image_size[1:3])	# 이미지 불러오기
        img = image.img_to_array(img)				# (height, width, channel) (224, 224, 3)
        img = img/255.	

        img = np.expand_dims(img, axis=0)	

        pred = cls._model.predict(img)
        pred_c = cls._model.predict_step(img)

        print('np.argmax(pred[0])\t\t{}'.format(np.argmax(pred[0])))
        print('index[np.argmax(pred[0])]\t{}'.format(cls._classes[np.argmax(pred[0])]))
        print('pred[0][np.argmax(pred[0])]\t{}\n'.format(pred[0][np.argmax(pred[0])]))

        return (np.argmax(pred[0]), cls._classes[np.argmax(pred[0])], pred[0][np.argmax(pred[0])])



# CNN 학습 코드
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from keras.layers import *
# from keras.models import *
# from keras.utils import *
# from sklearn.preprocessing import *
# import seaborn as sns

# from keras.datasets import cifar10

# (X_train, Y_train) , (X_test, Y_test) = cifar10.load_data()

# X_train.shape

# fig = plt.figure(figsize=(20,5))

# for i in range(36):
#     ax = fig.add_subplot(3, 12, i+1, xticks=[], yticks=[])
#     ax.imshow(X_train[i])

# X_train = X_train/255.0
# X_test = X_test /255.0


# Y_train = to_categorical(Y_train)
# Y_test = to_categorical(Y_test)


# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=4, padding='same', strides=1, activation='relu', input_shape=(32,32,3)))
# model.add(MaxPool2D(pool_size=2))
# model.add(Conv2D(filters=32, kernel_size=4, padding='same', strides=1, activation='relu'))
# model.add(MaxPool2D(pool_size=2))
# model.add(Conv2D(filters=64, kernel_size=4, padding='same', strides=1, activation='relu'))
# model.add(MaxPool2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.summary()


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# model.fit(X_train, Y_train, batch_size=150, epochs=1, validation_split=0.1)

# """
# Train on 45000 samples, validate on 5000 samples
# Epoch 1/1
# 45000/45000 [==============================] - 11s 248us/step - loss: 1.6029 - acc: 0.4202 - val_loss: 1.3092 - val_acc: 0.5284
# """


# score = model.evaluate(X_test, Y_test)
# print(score)

# """
# 10000/10000 [==============================] - 1s 78us/step
# [1.3261255556106568, 0.5187]
# """

# model.save('sesac_cnn2.h5')


# from tensorflow.keras.preprocessing import image
# model = load_model('sesac_cnn2.h5')

# index = ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']
# file_path = '../input/'
# files = os.listdir(file_path)
# image_size = (1, 32, 32, 3)


# img = image.load_img(file_path+files[0],target_size=image_size[1:3])	# 이미지 불러오기
# img = image.img_to_array(img)				# (height, width, channel) (224, 224, 3)
# img = img/255.	

# img = np.expand_dims(img, axis=0)	

# pred = model.predict(img)
# pred_c = model.predict_step(img)

# print('pred\t\t\t{}'.format(pred))
# print('pred_c\t\t\t{}\n'.format(pred_c))

# print('pred[0]\t\t{}'.format(pred[0]))
# print('pred_c[0]\t{}\n'.format(pred_c[0]))

# print('np.argmax(pred[0])\t\t{}'.format(np.argmax(pred[0])))
# print('index[np.argmax(pred[0])]\t{}'.format(index[np.argmax(pred[0])]))
# print('pred[0][np.argmax(pred[0])]\t{}\n'.format(pred[0][np.argmax(pred[0])]))

# # image = image.resize((32,32))
# # image = np.array(image) #convert image to numpy array
# # print(image.shape) #(32, 32, 3)
# # x_test = [image]

# # y_predict = model.predict(x_test)
# # print(y_predict) #[2]
# # print(labels[y_predict[0]]) #2
