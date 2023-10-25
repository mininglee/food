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
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

# 이미지를 저장할 기본 디렉토리 설정
data_dir = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb1/image/")
data_dir_test = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/test_rgb1/image/")

# 이미지 확장자 목록
valid_image_extensions = (".jpg", ".jpeg", ".png", ".gif")

# 이미지 파일 목록을 저장할 리스트 초기화
image_paths = []

# 데이터 디렉토리에서 이미지 파일 목록을 가져옴
for ext in valid_image_extensions:
    image_paths.extend(list(data_dir.glob(f"*/*{ext}")))
    image_paths.extend(list(data_dir_test.glob(f"*/*{ext}")))

# 이미지 개수 계산
image_count = len(image_paths)

# 이미지 크기 및 배치 크기 설정
batch_size = 512
img_height = 50
img_width = 50

# 데이터셋 생성
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="int"  # 레이블을 정수로 설정
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_test,
    validation_split=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="int"  # 레이블을 정수로 설정
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

# 레이블 데이터 추출
train_labels = []
val_labels = []

for image_path in image_paths:
    # 파일 경로에서 레이블 추출
    label = pathlib.Path(image_path).parent.name
    
    # train 또는 validation에 따라 레이블 데이터를 나눔
    if "learning_rgb1" in str(image_path):
        train_labels.append(label)
    else:
        val_labels.append(label)

train_labels = np.array(train_labels, dtype=np.object)
val_labels = np.array(val_labels, dtype=np.object)

num_classes = 300
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
    layers.Dense(num_classes, activation='softmax', name="outputs")
    ])

# 모델 컴파일 및 학습
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 다중 클래스 분류 문제의 손실 함수
              metrics=['accuracy'])

epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 모델 평가
model.evaluate(val_ds)

# 이미지 테스트
test_path = "C:/Users/KITCOOP/kicpython/hansik/kfood_test/test_rgb1/image/"

# 이미지 파일 읽어오기
image = tf.io.read_file(test_path)
image = tf.image.decode_image(image)  # 이미지 파일 형식 자동 감지

# 이미지 자동 사이즈 조정 및 예측
image_array = tf.expand_dims(image, 0)  # 배치 생성

predictions = model.predict(image_array)
scores = tf.nn.softmax(predictions[0])

for i in range(len(class_names)):
    print("클래스 '{}': {:.2f}% 확률로 속함".format(class_names[i], 100 * scores[i]))

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

# 이미지 테스트와 오류 처리 및 모델 모니터링 실행
handle_errors_and_monitoring(val_ds, model)
