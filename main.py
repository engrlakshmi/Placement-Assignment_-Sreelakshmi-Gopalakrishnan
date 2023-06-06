#Question 1 -
#Implement 3 different CNN architectures with a comparison table for the MNSIT
#dataset using the Tensorflow library.
#Note -
#1. The model parameters for each architecture should not be more than 8000
#parameters
#2. Code comments should be given for proper code understanding.
#3. The minimum accuracy for each accuracy should be at least 96%
import tensorflow as tf
from tensorflow.keras import layers
# Load and preprocess the MNIST dataset
import tensorflow as tf
from tensorflow.keras import layers

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

# Define the CNN architectures

# Model 1
model1 = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Model 2
model2 = tf.keras.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Model 3
model3 = tf.keras.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the models
models = [model1, model2, model3]
histories = []

for i, model in enumerate(models):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)
    histories.append(history)

# Compare the models' performance
table = "| Model | Parameters | Test Accuracy |\n"
table += "|-------|------------|---------------|\n"

for i, model in enumerate(models):
    num_params = model.count_params()
    test_acc = histories[i].history['val_accuracy'][-1] * 100
    table += f"| Model {i+1} | {num_params} | {test_acc:.2f}% |\n"

print(table)
