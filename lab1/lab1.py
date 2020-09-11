import numpy as np


def minkowski_vec(x1,x2,p=2.0):
    dist = np.sum(np.abs(x1-x2)**p)**1./p
    return dist

def minkowski_mat(x1, X2, p=2.0):
    dist = (np.sum(np.abs(x1-X2)**p, axis=1))**(1./p)
    return dist


def knn(x, data,p=2):
    features = data[:, :-1]
    distances = minkowski_mat(x, features,p=p)
    nn_index = np.argmin(distances)
    return data[nn_index, -1]

def iris_knn():
    iris = np.loadtxt("iris.txt")

    predictions = np.zeros(iris.shape[0])
    for i in range(iris.shape[0]):
        predictions[i] = knn(iris[i,:-1],iris)
        
        print("ground truth:" , iris[i,-1])
        print(predictions[i])

    targets = iris[:,-1]
    #$print("error rate:", (1.0==(predictions==targets).mean())*100.0)


def tests():

    iris_knn()

tests()
