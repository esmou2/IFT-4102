import numpy as np

class NeuralNet:

	def __init__(self, nHiddenLayers, nNeurons, inputSize, outputSize):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.nHiddenLayers = nHiddenLayers
		self.nNeurons = nNeurons

		self.layers = []
		if self.nHiddenLayers > 0:
			self.layers.append(NeuronLayer(self.inputSize, self.nNeurons))
			for i in range(self.nHiddenLayers - 1):
				self.layers.append(NeuronLayer(self.nNeurons, self.nNeurons))
		self.layers.append(NeuronLayer(self.nNeurons, self.outputSize))


	def train(self, train, train_labels):
		train_labels = np.array([[x] for x in train_labels], dtype=float)
		train = np.array(train)
		o = self.forward(train)
		self.backward(train, train_labels, o)

	def predict(self, exemple, label):
		o = exemple
		for l in self.layers:
			o = self.sigmoid(np.dot(o, l.synaptic_weights))
		return o[0]

	def test(self, test, test_labels):
		predictions = self.forward(test).T[0]
		print "Accuracy: ", getAccuracy(test_labels, predictions)
		return sum(test_labels - predictions)/float(len(predictions))
        # confusion_matrix = np.zeros((n,n))
        # for i in range(len(test)):
        #     confusion_matrix[test_labels[i], prediction] += 1
        # print "Confusion Matrix:"
        # print confusion_matrix

	def sigmoid(self, x,deriv=False):
		if(deriv==True):
			return x*(1-x)

		return 1/(1+np.exp(-x))

	def forward(self, X):
		self.zList = []
		lastz = self.sigmoid(np.dot(X, self.layers[0].synaptic_weights))
		self.zList.append(lastz)
		for w in self.layers[1:]:
			lastz = self.sigmoid(np.dot(lastz, w.synaptic_weights))
			self.zList.append(lastz)

		return self.zList[-1]

	def backward(self, X, y, o):
		reversedzList = list(reversed(self.zList))
		reversedLayersList = list(reversed(self.layers))
		error = y - o
		delta = error*self.sigmoid(o, True)

		deltas = []
		deltas.append(delta)

		for i in range(len(self.zList) - 1):
			error = delta.dot(reversedLayersList[i].synaptic_weights.T)
			delta = error*self.sigmoid(reversedzList[i+1], True)
			deltas.append(delta)

		deltas = list(reversed(deltas))

		self.layers[0].synaptic_weights += X.T.dot(deltas[0])
		for i in range(1, len(self.layers)):
			self.layers[i].synaptic_weights += self.zList[i - 1].T.dot(deltas[i])

def getAccuracy(labels, predictions):
	correct = 0
	for x in range(len(labels)):
		if labels[x] - predictions[x] < 1e-10:
			correct += 1
	return (correct/float(len(labels))) * 100.0

class NeuronLayer():
    def __init__(self, number_of_inputs_per_neuron, number_of_neurons):
        self.synaptic_weights = np.random.randn(number_of_inputs_per_neuron, number_of_neurons)

    def __repr__(self):
        return str(self.synaptic_weights)