import matplotlib.pyplot as plt
import numpy as np
import keras

# from tensorflow.keras.datasets import cifar100  # Using Keras for loading

cifar100=keras.datasets.cifar100

# Load the CIFAR-100 dataset
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

# Load class names
_, class_names = cifar100.load_data(label_mode='fine')

# Number of images to display
num_images = 10

# Create a figure to display images
plt.figure(figsize=(15, 5))

for i in range(num_images):
    plt.subplot(2, 5, i + 1)  # Arrange images in a 2x5 grid
    plt.imshow(train_images[i])
    plt.title(f"Label: {class_names[train_labels[i]]}")  # Accessing class name
    plt.axis('off')

plt.tight_layout()
plt.show()