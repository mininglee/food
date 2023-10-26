# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:18:06 2023

@author: user
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib

# 데이터 경로 설정
data_dir = pathlib.Path("C:/Users/koung/kicpython/hansik/kfood_new/learning_rgb1/image/")
data_dir_test = pathlib.Path("C:/Users/koung/kicpython/hansik/kfood_new/test_rgb1/image/")

# 이미지 크기 및 배치 크기 설정
img_height = 50
img_width = 50
batch_size = 32

# 데이터셋 생성을 위한 ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# 검증 데이터에는 데이터 증강을 적용하지 않습니다.
val_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 불러오기
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # 다중 클래스 분류 문제
    shuffle=True)

validation_generator = val_datagen.flow_from_directory(
    data_dir_test,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # 다중 클래스 분류 문제
    shuffle=False)  # 검증 데이터는 섞지 않음

# 모델 생성
num_classes = len(train_generator.class_indices)
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # 다중 클래스 분류 문제의 손실 함수
              metrics=['accuracy'])

epochs = 20
history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

# 모델 평가
model.evaluate(validation_generator)

# 이미지 테스트 함수 정의
def test_image(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # 모델의 입력 크기에 맞게 차원 확장
    predictions = model.predict(img_array)
    predicted_class = train_generator.class_indices
    inv_map = {v: k for k, v in predicted_class.items()}
    predicted_class = inv_map[np.argmax(predictions)]
    return predicted_class

# 이미지 테스트 및 예측
test_path = "C:/Users/koung/kicpython/hansik/kfood_new/test_rgb1/model1/"
result = test_image(model, test_path)
print(f"예측 클래스: {result}")
