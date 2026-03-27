from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from sklearn . datasets import load_breast_cancer
data = load_breast_cancer ()

tf.random.set_seed(1)

features_x = data.data
target_y = data.target

mu = features_x.mean(axis=(0,1))
sigma = features_x.std(axis=(0,1))
x_std = (features_x - mu) / (sigma + 1e-8)

features_train, features_test, labels_train, labels_test = train_test_split(x_std, target_y, test_size=0.20,random_state=42)

model = Sequential()
input_layer = InputLayer(input_shape=(30,))
model.add(input_layer)
hidden_layer = Dense(10)
model.add(hidden_layer)
output_layer = Dense(1, activation='sigmoid')
model.add(output_layer)

# compile and train
model.compile(loss="binary_crossentropy", metrics=["accuracy"] )
history = model.fit(x_std, target_y, epochs = 10)

test_accuracy = model.evaluate(features_test, labels_test)
train_accuracy = model.evaluate(features_train, labels_train)
print(f"Test accuracy: {test_accuracy[1]}")
print(f"Train accuracy: {train_accuracy[1]}")

"""
importance of feature scaling in neural networks:
scaling makes sure the inputs are in a consistent numeric range, which helps the model learn more effectively

an epoch is one complete pass through the data. the model can run its computations and predictions, and update it's weights to improve its accuracy.
"""
