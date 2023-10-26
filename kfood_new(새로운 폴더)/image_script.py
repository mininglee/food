import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 모델을 불러오는 코드
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

# 이미지를 예측하는 함수
def predict_image(model, img_path, img_height, img_width):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class