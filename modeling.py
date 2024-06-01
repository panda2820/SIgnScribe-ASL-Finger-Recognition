import sys
import io
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Set default encoding to 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load Sign Language MNIST dataset
def load_sign_language_mnist(path):
    train = pd.read_csv(os.path.join(path, 'sign_mnist_train.csv'))
    test = pd.read_csv(os.path.join(path, 'sign_mnist_test.csv'))
    
    train_labels = train['label'].values
    train_images = train.drop('label', axis=1).values
    test_labels = test['label'].values
    test_images = test.drop('label', axis=1).values
    
    # Reshape images to 28x28 pixels
    train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))
    
    # Normalize pixel values
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # One-hot encode labels
    train_labels = tf.keras.utils.to_categorical(train_labels, 25)
    test_labels = tf.keras.utils.to_categorical(test_labels, 25)
    
    return (train_images, train_labels), (test_images, test_labels)

# Path to the dataset
path = 'E:/Fyp/new'

(train_images, train_labels), (test_images, test_labels) = load_sign_language_mnist(path)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(25, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# model.save('asl_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

model.summary()