from sklearn . datasets import load_breast_cancer
data = load_breast_cancer ()

features_x = data.data
target_y = data.target

print(f"shape of X: {features_x.shape}")
print(f"shape of y: {target_y.shape}")

samples0 = 0
samples1 = 0

for row in target_y:
    if row == 0:
        samples0 += 1
    else:
        samples1 += 1
print(f"{samples0} samples are class 0.")
print(f"{samples1} samples are class 1.")

"""
This data set is imbalanced. There are 212 samples classified as 0, and 357 samples classified as 1.
It is important to have class balance in a dataset so the model can learn equally from each classificiation.
A balanced dataset can lead to higher model accuracy and better model performance.
"""
