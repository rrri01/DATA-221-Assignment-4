import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import layers, models

from sklearn . datasets import load_breast_cancer
data = load_breast_cancer ()

tf.random.set_seed(1)

features_x = data.data
target_y = data.target

mu = features_x.mean(axis=(0,1))
sigma = features_x.std(axis=(0,1))
x_std = (features_x - mu) / (sigma + 1e-8)

features_train, features_test, labels_train, labels_test = train_test_split(x_std, target_y, test_size=0.20,random_state=42)

# neural network:
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

neural_predict = model.predict(features_test)

neural_predicted = np.round(neural_predict).tolist()
neural_confusion_matrix = metrics.confusion_matrix(labels_test, neural_predicted)
print(f"neural network:\n{neural_confusion_matrix}")

# deicion tree:
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=8)
decision_tree_classifier.fit(features_train, labels_train)

test_predicted_labels = decision_tree_classifier.predict(features_test)
accuracy = accuracy_score(labels_test, test_predicted_labels)

train_predicted_labels = decision_tree_classifier.predict(features_train)
train_accuracy = accuracy_score(labels_train, train_predicted_labels)

decision_matrix = metrics.confusion_matrix(labels_test, test_predicted_labels)
print(f"decision tree: \n{decision_matrix}")

"""
i think the neural network would be slightly better, since there are no false negatives.
since this model is detecting breast cancer, there are more negative effects associated with not being diagnosed, compared to being wrongly diagnosed.

neural network:
- advantage: there are no false negatives for this dataset
- limitation: there are more false positives than the decision tree

decision tree:
- advantage: more positives were correctly identified than the neural network
- limitation: there was a false negative, aka the patient was diagnosed as not having cancer when they did, which could have extreme consequences.

"""