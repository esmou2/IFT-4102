import load_datasets
import numpy as np
import DecisionTree

n= 1;

train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(0.3)
train_votes, train_labels_votes, test_votes, test_labels_votes = load_datasets.load_congressional_dataset(0.5)
train_monks, train_labels_monks, test_monks, test_labels_monks = load_datasets.load_monks_dataset(n)


x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
y = np.array([0, 0, 0, 1, 1, 0])

# print np.array([x1, x2]).T
# print y

# print(np.array(train_iris).T)
# print (np.array(train_labels_iris))
#
# print train_iris
# print train_labels_iris

# print train_iris

dt = DecisionTree.DecisionTree();
dt.train(train_iris, train_labels_iris)
# print train_iris
dt.test(test_iris, test_labels_iris)
# print dt.predict([3.0, 3.0, 2.0, 2.0, 3.0, 1.0], 1)
# dt.train(np.array([x1, x2]).T, y)


