import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers

img_dir = './train/kagle_dataset'
train_data = tf.keras.preprocessing.image_dataset_from_directory(img_dir,                                                                  
    label_mode = "categorical",
    image_size = (224,224),
    batch_size= 32,
    seed = 42,
    validation_split = 0.25,
    subset = "training"
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(img_dir,
    label_mode = "categorical",
    image_size = (224,224),
    batch_size = 32,
    seed = 42,
    validation_split = 0.25,
    subset = "validation"
)

model = tf.keras.Sequential([
    layers.Conv2D(filters = 20, kernel_size = 2,input_shape= (224,224,3),padding="same",activation= "relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size= 2),
    layers.Conv2D(filters = 20, kernel_size = 2, padding= "same",activation= "relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size= 2),
    layers.Conv2D(filters = 20, kernel_size= 2, padding = "same",activation= "relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size = 2),
    layers.Conv2D(filters = 20, kernel_size= 2, padding = "same",activation= "relu"),
    layers.Flatten(),
    layers.Dense(1024, activation = "relu"),                        
    layers.Dropout(0.3),
    layers.Dense(1024, activation = "relu"),                          
    layers.Dropout(0.3),
    layers.Dense(1024, activation = "relu"),
    layers.Dense(53, activation= "softmax")
])

model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001),
    metrics = ["accuracy"]
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./results/second_run_64_64.h5', verbose=1, save_weights_only=False, save_freq='epoch')

model.fit(
    train_data,
    steps_per_epoch = len(train_data),      
    epochs = 200,
    validation_data = test_data,
    validation_steps = len(test_data),
    callbacks = [model_checkpoint]    
)
