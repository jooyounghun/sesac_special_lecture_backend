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

