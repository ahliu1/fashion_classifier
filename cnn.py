import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Making sure images are scaled between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0

# Changing dimension for CNN
train_images = train_images.reshape(-1, 28, 28, 1) # -1: # of images, 28: width, 28: height, 1: # channels (grayscale)
test_images = test_images.reshape(-1, 28, 28, 1)

# Build Model
model = tf.keras.Sequential([
    # first convolutional layer, 32 filters (convention)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # downsample the feature maps, take max in each 2x2 patch
    # reduce overfitting
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # flattens to 1d array (784 pixels)
    tf.keras.layers.Flatten(),

    # Add layer with 128 neurons to the model
    # RELU- introduces non-linearity
    tf.keras.layers.Dense(128, activation='relu'),

    # output layer, 10 neurons (10 classes)
    tf.keras.layers.Dense(10)
])

# Compile Model
model.compile(optimizer='adam',
              # computationally efficient + has little memory requirement
              
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # minimize this
              
              metrics=['accuracy']
              # monitor training + testing steps
             )

# Fit Model
history = model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels))

# Results
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
training_accuracy = history.history['accuracy']

# Assess overfitting
print('\n Training accuracy and testing accuracy difference is:', train_accuracy[-1] - test_acc)

# Please see results and plots in Jupyter notebook.