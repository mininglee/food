import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib

# 데이터 경로 설정
data_dir = pathlib.Path("C:/Users/koung/kicpython/hansik/kfood_new/data/learning_rgb1/image/")

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

# 데이터 불러오기
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # 다중 클래스 분류 문제
    shuffle=True)

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
history = model.fit(train_generator, epochs=epochs)

# 모델 저장
model.save("C:/Users/koung/kicpython/hansik/kfood_new/models/model.h5")