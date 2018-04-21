import load_datasets
import numpy as np
import DecisionTree

n= 1;

train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(0.2)


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
# dt.train(train_iris, train_labels_iris)
dt.train(np.array([x1, x2]).T, y)
