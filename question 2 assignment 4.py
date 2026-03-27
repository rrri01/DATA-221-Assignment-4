from sklearn . datasets import load_breast_cancer
data = load_breast_cancer ()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

features_x = data.data
target_y = data.target

features_train, features_test, labels_train, labels_test = train_test_split(features_x, target_y, test_size=0.20,random_state=42)

decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(features_train, labels_train)

test_predicted_labels = decision_tree_classifier.predict(features_test)
accuracy = accuracy_score(labels_test, test_predicted_labels)


train_predicted_labels = decision_tree_classifier.predict(features_train)
train_accuracy = accuracy_score(labels_train, train_predicted_labels)

print(f"test accuracy: {accuracy}")
print(f"train accuracy: {train_accuracy}")


"""
In decision trees, entropy represents how pure a partition is. A lower entropy means a better split.

The testing accuracy (94.68%) is pretty high which shows good generalization.
However, the training accuracy is 100%, which suggests overfitting on the data.

"""

