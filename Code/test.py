import sys
import Knn
import load_datasets
import numpy as np
from collections import Counter, defaultdict

def occurrences(labels):
		no_of_examples = len(labels)
		prob = dict(Counter(labels))
		for key in prob.keys():
			prob[key] = prob[key] / float(no_of_examples)
		return prob


train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.1)

knn = Knn.Knn(5)
conversion_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


train = np.array(train)

classes     = list(set(train_labels))
rows, cols  = np.shape(train)
likelihoods = {}
for cls in classes:
    likelihoods[cls] = defaultdict(list)

class_probabilities = occurrences(train_labels)

for cls in classes:
    row_indices = []
    for i in range(len(train_labels)):
        if train_labels[i] == cls:
            row_indices.append(i)
    subset      = train[row_indices, :]
    r, c        = np.shape(subset)
    for j in range(c):
        likelihoods[cls][j] += list(subset[:,j])

for cls in classes:
    for j in range(0, c):
        likelihoods[cls][j] = occurrences(likelihoods[cls][j])
new_sample = [5.1,3.5,1.4,0.2]
results = {}
for cls in classes:
    class_probability = class_probabilities[cls]
    for i in range(len(new_sample)):
        relative_feature_values = likelihoods[cls][i]
        if new_sample[i] in relative_feature_values.keys():
            class_probability *= relative_feature_values[new_sample[i]]
        else:
            class_probability *= 0
        results[cls] = class_probability

print results
