import numpy as np
from pprint import pprint

class DecisionTree:

    def __init__(self):
        self.tree = {}

    def train(self, train, train_labels):
        self.tree = recursive_split(np.array(train), np.array(train_labels))
        pprint(self.tree)

    def predict(self, exemple, label):
        print "predict"

	def test(self, test, test_labels):
		print "test"


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def mutual_information(y, x):
    res = entropy(y)
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

def recursive_split(x, y):
    if len(set(y)) == 1 or len(y) == 0:
        return y

    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])

    selected_attr = np.argmax(gain)

    sets = partition(x[:, selected_attr])
    set_test = set()
    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        res["x_{} = {}".format(selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res




