# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:18:06 2023

@author: user
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

# 데이터 디렉토리 설정
data_dir = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb1/image/")
data_dir_test = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/test_rgb1/")

image_count = len(list(data_dir.glob('*/*.jpg')))
image_test_count = len(list(data_dir_test.glob('*/*.jpg')))

# 이미지 크기 및 배치 크기 설정
batch_size = 512
img_height = 50
img_width = 50

# 데이터셋 생성
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=None,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_test,
    validation_split=None,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 이미지 클래스 이름 확인
class_names = train_ds.class_names
print(class_names)

# 성능 개선을 위한 데이터셋 전처리 및 캐싱
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 데이터 정규화 레이어 설정
normalization_layer = layers.Rescaling(1./255)

# 데이터 증강 설정
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

num_classes = 300
# CNN 모델 정의
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 모델 평가
model.evaluate(val_ds)

# 이미지 테스트
test_path = "C:/Users/KITCOOP/kicpython/hansik/kfood_test/test_rgb1/"
Image.open(test_path)

# 이미지 자동 사이징 및 예측
img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # 배치 생성

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("이미지는 {} with a {:.2f} percent 확신합니다.".format(class_names[np.argmax(score)], 100 * np.max(score)))

# 예측이 어려운 데이터 처리 및 기록 및 로깅
def process_uncertain_prediction(prediction, threshold=0.6):
    if prediction.max() < threshold:
        print("모델 확신도 부족, 인간 검토 필요")
    return prediction.argmax()

# 오류 처리 및 모델 모니터링
def handle_errors_and_monitoring(data, model):
    for batch, labels in data:
        predictions = model.predict(batch)
        for i, prediction in enumerate(predictions):
            processed_result = process_uncertain_prediction(prediction)
            print(f"예측: {processed_result}, 실제 레이블: {labels[i]}")

# 오류 처리 및 모델 모니터링 실행
handle_errors_and_monitoring(val_ds, model)