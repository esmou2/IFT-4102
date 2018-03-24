import sys
import BayesNaif
import load_datasets
import numpy as np
from collections import Counter, defaultdict

def occurrences(labels):
		no_of_examples = len(labels)
		prob = dict(Counter(labels))
		for key in prob.keys():
			prob[key] = prob[key] / float(no_of_examples)
		return prob


train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.5)

b = BayesNaif.BayesNaif()
new_sample = [6.7,3.1,4.7,1.5]

b.train(train, train_labels)
b.test(test, test_labels)

