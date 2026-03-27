from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow . keras . datasets import fashion_mnist
( X_train , y_train ) , ( X_test , y_test ) = fashion_mnist . load_data ()


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



probs = model.predict(X_test, verbose=0)

pred_labels = []

for x in probs:
    y = int(tf.argmax(x))
    pred_labels.append(y)

confusion_matrix = metrics.confusion_matrix(y_test, pred_labels)
print(confusion_matrix)

# misclassified:
wrong = 0
for i in range(0,len(pred_labels)):
    if pred_labels[i] != y_test[i]:
        plt.figure(figsize=[5, 5])
        plt.gray()
        plt.imshow(X_test[i,:,:], cmap='gray')
        plt.show()
        print(f"Predicted: {pred_labels[i]}, Actual: {y_test[i]})")
        wrong += 1
    if wrong == 3:
        break

"""
1. predicted 5, actual 9
- the model guessed it was a sandal when it was an ankle boot

2. predicted 0, actual 9
- the model guessed tshirt, the right answer was just a shirt

3. predicted 4, actual 6
- the model guessed coat when it was a shirt

a pattern in the misclassifications is that the model guesses something within the same category.
- sandals and ankle boots are both footwear
- tshirts, coats, and shirts are all shirts

one method to improve CNN performance:
- have more images to train the model so it can better distinguish between similar clothing items
- adding more epochs can also improve performance to some extent
"""
