import numpy as np
# -*- coding: utf-8 -*-

class Knn:

	def __init__(self, k, **kwargs):
		self.k = k


	def train(self, train, train_labels, x): #vous pouvez rajouter d'autres attribus au besoin
		labels = list(set(train_labels))
		n = len(labels)

		data_size = train.shape[0]

		neighbors_indexes = getKNearestNeighbor(x[:-1], self.k, train)

		self.test(train, train_labels)

		return getPrediction(neighbors_indexes)



	def predict(self, exemple, label):
		labels = list(set(exemple))
		votes = [0] * len(labels)
		for n in range(len(exemple)):
			for i in range(len(labels)):
				if exemple[n] == labels[i]:
					votes[i] += 1
		return labels[votes.index(max(votes))] == label


	def test(self, test, test_labels):
		error_count = 0

		n = len(set(test_labels))
		confusion_matrix = np.zeros((n,n))
		nb_attribution = [0] * n
		nb_correct_attribution = [0] * n
		nb_class_element = [0] * n

		for i in range(len(test)):
			prediction = self.train(test, test_labels, test[i,:])
			if prediction != test_labels[i]:
				error_count += 1
			else:
				nb_correct_attribution[test_labels[i]] += 1
			confusion_matrix[test_labels[i], prediction] += 1
			nb_attribution[prediction] += 1
			nb_class_element[test_labels[i]] += 1

		precision = []
		recall = []
		for i in len(n):
			precision.append(nb_correct_attribution[i] / nb_attribution[i])
			recall.append(nb_correct_attribution[i] / nb_class_element[i])


		print confusion_matrix


def distance (a, b):
    if (len (a) != len (b)):
        return (-1)
    dist = 0
    for i in range (len (a)):
        dist += (a[i] - b[i])
    return dist

def getKNearestNeighbor (x, k, train):
    distances = []
    for i in range(len(train)):
        distances.append(distance(x, train[i])) # Calculer toutes les distances
    knn = []
    for i in range(k):
        p = float("inf")
        for j in range(len(train)):
            if distances[j] != 0 and distances[j] < p and j not in knn:
                p = distances[j]
                indice = j
        knn.append(indice)
    return knn

def getPrediction(neighbors_label):
    labels = list(set(neighbors_label))
    votes = [0] * len(labels)
    for n in range(len(neighbors_label)):
        for i in range(len(labels)):
            if neighbors_label[n] == labels[i]:
                votes[i] += 1
    return labels[votes.index(max(votes))]

def getAccuracy(labels, predictions):
    correct = 0
    for x in range(len(labels)):
        if labels[x] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0