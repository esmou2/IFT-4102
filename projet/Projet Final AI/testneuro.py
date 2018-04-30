import numpy as np
import copy

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

X = np.array(([0,0,1], [0,1,1], [1,0,1], [1,1,1]), dtype=float)

y = np.array(([0], [1], [1], [0]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
# y = y/100 # max test score is 100

class Neural_Network(object):
    def __init__(self, nHiddenLayers, nNeurons, inputSize, outputSize):
        #parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.nHiddenLayers = nHiddenLayers
        self.nNeurons = nNeurons

        #weights
        self.layers = []
        if self.nHiddenLayers > 0:
            self.layers.append(NeuronLayer(self.inputSize, self.nNeurons))
            for i in range(self.nHiddenLayers - 1):
                self.layers.append(NeuronLayer(self.nNeurons, self.nNeurons))
        self.layers.append(NeuronLayer(self.nNeurons, self.outputSize))

    def forward(self, X):

        self.zList = []
        lastz = self.sigmoid(np.dot(X, self.layers[0].synaptic_weights))
        self.zList.append(lastz)
        for w in self.layers[1:]:
            lastz = self.sigmoid(np.dot(lastz, w.synaptic_weights))
            self.zList.append(lastz)

        return self.zList[-1]

    def sigmoid(self, x,deriv=False):
        if(deriv==True):
            return x*(1-x)

        return 1/(1+np.exp(-x))

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


    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

class NeuronLayer():
    def __init__(self, number_of_inputs_per_neuron, number_of_neurons):
        self.synaptic_weights = np.random.randn(number_of_inputs_per_neuron, number_of_neurons)

    def __repr__(self):
        return str(self.synaptic_weights)

NN = Neural_Network(1, 10, 3, 1)
for i in xrange(1000):
    NN.train(X, y)

print "Actual Output: \n" + str(y)
print "Predicted Output: \n" + str(NN.forward(X))