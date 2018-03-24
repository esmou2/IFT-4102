import numpy as np
from collections import Counter, defaultdict

class BayesNaif:

	def __init__(self, **kwargs):
		pass


	def train(self, train, train_labels):
		train = np.array(train)

		self.classes = list(set(train_labels))
		n, m  = np.shape(train)

		self.probabilities = get_prior_probabilies(train_labels)

		self.attributes_probabilities = {}
		for cls in self.classes:
			self.attributes_probabilities[cls] = defaultdict(list)

		for cls in self.classes:
			indexes = []
			for i in range(n):
				if train_labels[i] == cls:
					indexes .append(i)
			subset = train[indexes , :]
			r, c = np.shape(subset)
			for i in range(c):
				self.attributes_probabilities[cls][i] += list(subset[:,i])

		for cls in self.classes:
			for i in range(m):
				self.attributes_probabilities[cls][i] = get_prior_probabilies(self.attributes_probabilities[cls][i])

		self.test(train, train_labels)

	def predict(self, exemple, label):
		results = {}

		for cls in self.classes:
			class_probability = self.probabilities[cls]
			for i in range(len(exemple)):
				relative_attribute_values = self.attributes_probabilities[cls][i]
				if exemple[i] in relative_attribute_values.keys():
					class_probability *= relative_attribute_values[exemple[i]]
				else:
					class_probability *= 0
				results[cls] = class_probability

		return max(results, key=lambda i: results[i])

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



def get_prior_probabilies(labels):
	no_of_examples = len(labels)
	prob = dict(Counter(labels))
	for key in prob.keys():
		prob[key] = prob[key] / float(no_of_examples)
	return prob


def getAccuracy(labels, predictions):
    correct = 0
    for x in range(len(labels)):
        if labels[x] is predictions[x]:
            correct += 1
    return (correct/float(len(labels))) * 100.0