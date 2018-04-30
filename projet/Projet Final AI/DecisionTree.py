import numpy as np
from pprint import pprint
from collections import Counter

class DecisionTree:

    def __init__(self):
        self.tree = {}
        self.average = []

    def train(self, train, train_labels):
        self.tree = recursive_split(np.array(train), np.array(train_labels))
        for a in np.array(train).T:
            cnt = Counter(a)
            self.average.append(cnt.most_common(1)[0][0])
        print "=== test train data ==="
        self.test(train, train_labels)

    def predict(self, exemple, label):
        sub_tree = self.tree
        while type(sub_tree) is dict:
            att = sub_tree.keys()[0].attribute
            att_val = exemple[att]
            if attribute(att, att_val) in sub_tree.keys():
                sub_tree = sub_tree[attribute(att, att_val)]
            else:
                sub_tree = sub_tree[attribute(att, self.average[att])]
        return sub_tree[0]

    def test(self, test, test_labels):
        n = len(set(test_labels))
        confusion_matrix = np.zeros((n,n))
        predictions = [] # predicted classes
        for i in range(len(test)):
            prediction = self.predict(test[i], test_labels[i])
            predictions.append(prediction)
            confusion_matrix[test_labels[i], prediction] += 1
        print "Accuracy:", getAccuracy(test_labels, predictions)
        print "Confusion Matrix:"
        print confusion_matrix

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

    res = {}

    test = set()
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        result = recursive_split(x_subset, y_subset)
        res[attribute(selected_attr, k)] = result

    return res

def getAccuracy(labels, predictions):
    correct = 0
    for x in range(len(labels)):
        if labels[x] is predictions[x]:
            correct += 1
    return (correct/float(len(labels))) * 100.0


class attribute:
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

    def __hash__(self):
        return hash((self.attribute, self.value))

    def __eq__(self, other):
        return (self.attribute, self.value) == (other.attribute, other.value)

    def __repr__(self):
        return "{} = {}".format(self.attribute, self.value)
