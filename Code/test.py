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


train, train_labels, test, test_labels = load_datasets.load_congressional_dataset(1)

knn = Knn.Knn(5)
conversion_labels = ['republican','democrat']
new_sample = [0,2,0,1,1,1,0,0,0,0,0,1,1,1,0,0]

print conversion_labels[knn.train(train, train_labels, new_sample )]


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
    for j in range(cols):
        likelihoods[cls][j] = occurrences(likelihoods[cls][j])


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

print conversion_labels[max(results, key=lambda i: results[i])]
