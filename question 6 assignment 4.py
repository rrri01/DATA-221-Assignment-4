from sklearn import metrics
from tensorflow . keras . datasets import fashion_mnist
( X_train , y_train ) , ( X_test , y_test ) = fashion_mnist . load_data ()

# normalize:
mu = X_train.mean(axis=(0,1))
sigma = X_train.std(axis=(0,1))
x_std = (X_train - mu) / (sigma + 1e-8)

H, W, C = 28, 28, 3

import tensorflow as tf
from tensorflow.keras import layers, models

X_train = X_train.astype("float32") / 255.0
X_train = X_train.astype("float32") / 255.0

x_train = X_train[..., None]
x_test = X_train[..., None]

print("train shape:", x_train.shape)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
])
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

# compile and train
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"] )
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=15, batch_size=64)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("test loss:", test_loss)
print("test acc:", test_acc)


"""
a regular neural network would have too many parameters, which means the model takes up more memory, training is slower, and there is higher risk of overfitting

convolution layers apply filters to the input to obtain features, and patterns in images.
"""

