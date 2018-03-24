import numpy as np
from collections import Counter, defaultdict

class Knn:

	def __init__(self, k, **kwargs):
		self.k = k


	def train(self, train, train_labels): #vous pouvez rajouter d'autres attribus au besoin

		self.train_data = train
		self.train_data_labels = train_labels
		print "Test on training data: "
		self.test(train, train_labels)



	def predict(self, exemple, label):
		neighbors_indexes = getKNearestNeighbor(exemple, self.k, self.train_data)
		return getPrediction(neighbors_indexes, self.train_data_labels)


	def test(self, test, test_labels):
		n = len(set(test_labels))
		confusion_matrix = np.zeros((n,n))
		nb_attribution = [0] * n # number of attribution to the class i
		nb_correct_attribution = [0] * n # number of correct attribution to the class i
		nb_class_element = [0] * n # number of element in the class i
		predictions = [] # predicted classes

		for i in range(len(test)):
			prediction = self.predict(test[i], test_labels[i])
			predictions.append(prediction)
			if prediction == test_labels[i]:
				nb_correct_attribution[test_labels[i]] += 1
			confusion_matrix[test_labels[i], prediction] += 1 # real, predicted
			nb_attribution[prediction] += 1

		no_classes = dict(Counter(test_labels))
		precision = []
		recall = []

		for i in range(n):
			precision.append(nb_correct_attribution[i] / float(no_classes[i]))
			recall.append(nb_correct_attribution[i] / float(no_classes[i]))

		print "Accuracy:", getAccuracy(test_labels, predictions)
		print "Precision: ", precision
		print "Recall: ", recall
		print "Confusion Matrix:"
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
	classes = list(set(neighbors_labels))
	votes = [0] * len(classes)
	for n in neighbors_indexes:
		for i in range(len(classes)):
			if neighbors_labels[n] == classes[i]:
				votes[i] += 1
	return classes[votes.index(max(votes))]

def getAccuracy(labels, predictions):
    correct = 0
    for x in range(len(labels)):
        if labels[x] is predictions[x]:
            correct += 1
    return (correct/float(len(labels))) * 100.0