import numpy as np
from collections import Counter, defaultdict

class Knn:

	def __init__(self, k, **kwargs):
		self.k = k


	def train(self, train, train_labels, x): #vous pouvez rajouter d'autres attribus au besoin
		labels = list(set(train_labels))
		n = len(labels)
		data_size = np.array(train).shape[0]

		neighbors_indexes = getKNearestNeighbor(x, self.k, train)

		return getPrediction(neighbors_indexes, train_labels)



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
			prediction = self.train(test, test_labels, test[i])
			if prediction != test_labels[i]:
				error_count += 1
			else:
				nb_correct_attribution[test_labels[i]] += 1
			confusion_matrix[test_labels[i], prediction] += 1
			nb_attribution[prediction] += 1

		no_classes = dict(Counter(test_labels))
		precision = []
		recall = []

		print error_count
		for i in range(n):
			precision.append(nb_correct_attribution[i] / float(no_classes[i]))
			recall.append(nb_correct_attribution[i] / float(no_classes[i]))
		print "precision: ", precision
		print "recall: ", recall
		print confusion_matrix


def distance (a, b):
	if (len(a) != len(b)):
		return (-1)
	dist = 0
	for i in range (len(a)):
		dist += (a[i] - b[i])**2
	return dist**0.5

def getKNearestNeighbor(x, k, train):
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

def getPrediction(neighbors_indexes, neighbors_labels):
	labels = []
	for i in neighbors_indexes:
		labels.append(neighbors_labels[i])
	labels = list(set(labels))
	votes = [0] * len(labels)
	for n in neighbors_indexes:
		for i in range(len(labels)):
			if neighbors_labels[n] == labels[i]:
				votes[i] += 1
	return labels[votes.index(max(votes))]

def getAccuracy(labels, predictions):
    correct = 0
    for x in range(len(labels)):
        if labels[x] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0