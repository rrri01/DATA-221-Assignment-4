from sklearn . datasets import load_breast_cancer
data = load_breast_cancer ()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

features_x = data.data
target_y = data.target

features_train, features_test, labels_train, labels_test = train_test_split(features_x, target_y, test_size=0.20,random_state=42)

decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=8)
decision_tree_classifier.fit(features_train, labels_train)

test_predicted_labels = decision_tree_classifier.predict(features_test)
accuracy = accuracy_score(labels_test, test_predicted_labels)


train_predicted_labels = decision_tree_classifier.predict(features_train)
train_accuracy = accuracy_score(labels_train, train_predicted_labels)

print(f"test accuracy: {accuracy}")
print(f"train accuracy: {train_accuracy}")

print(decision_tree_classifier.feature_importances_)

"""
controlling model complexity can reduce overfitting because:
- if the model is too complex, then it will perform well on the training data, but will be too specific to one dataset to perform well on a new dataset.
- if we simplify the model, there will be less parameters, and will perform better on entirely new data.

- feature importance tells you which features of the dataset are more useful for predicting the outcome.
- it contributes to the interpretability because it can help determine how complex or simple the tree is.

"""

