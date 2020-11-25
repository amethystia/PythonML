#导入package和相关绘图参数
import random
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import Counter

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i, j] = np.linalg.norm(self.X_train[j, :] - X[i, :])
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i, :] = np.linalg.norm(self.X_train - X[i, :], axis=1)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists
#定义L2距离度量函数
  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    M = np.dot(X, self.X_train.T)
    te = np.square(X).sum(axis=1)
    tr = np.square(self.X_train).sum(axis=1)
    dists = np.sqrt(-2 * M + tr + np.matrix(te).T)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
#使用多数表决的分类决策规则定义预测函数，假设K取1
  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      labels = self.y_train[np.argsort(dists[i, :])].flatten()
      closest_y = labels[0:k]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      c = Counter(closest_y)
      y_pred[i] = c.most_common(1)[0][0]
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred
# 使用5折交叉验证来选择最优的K值
  def cross_validation(self, X_train, y_train):
      num_folds = 5
      k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

      X_train_folds = []
      y_train_folds = []

      X_train_folds = np.array_split(X_train, num_folds)
      y_train_folds = np.array_split(y_train, num_folds)

      k_to_accuracies = {}

      for k in k_choices:
        for fold in range(num_folds):  # This fold will be omitted.
          # Creating validation data and temp training data
          validation_X_test = X_train_folds[fold]
          validation_y_test = y_train_folds[fold]
          temp_X_train = np.concatenate(X_train_folds[:fold] + X_train_folds[fold + 1:])
          temp_y_train = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])

          # Initializing a class
          self.train(temp_X_train, temp_y_train)

          # Computing the distance
          temp_dists = self.compute_distances_no_loops(validation_X_test)
          temp_y_test_pred = self.predict_labels(temp_dists, k=k)
          temp_y_test_pred = temp_y_test_pred.reshape((-1, 1))
          # Checking accuracies
          num_correct = np.sum(temp_y_test_pred == validation_y_test)
          num_test = validation_X_test.shape[0]
          accuracy = float(num_correct) / num_test
          k_to_accuracies[k] = k_to_accuracies.get(k, []) + [accuracy]

      # Print out the computed accuracies
      for k in sorted(k_to_accuracies):
        accuracies=k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

        for accuracy in k_to_accuracies[k]:

          print('k = %d, accuracy = %f' % (k, accuracy))

      accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
      best_k = k_choices[np.argmax(accuracies_mean)]
      print('最佳k值为{}'.format(best_k))
      accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
      plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
      plt.title('Cross-validation on k')
      plt.xlabel('k')
      plt.ylabel('Cross-validation accuracy')
      plt.show()
      return best_k

  def create_train_test(self):
    iris = datasets.load_iris()
    X, y = shuffle(iris.data, iris.target, random_state=13)
    X = X.astype(np.float32)
    y = y.reshape((-1, 1))
    #训练集和测试集的简单划分
    offset = int(X.shape[0] * 0.7)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    return X_train, y_train, X_test, y_test




if __name__ == '__main__':
    knn_classifier = KNearestNeighbor()
    X_train, y_train, X_test, y_test = knn_classifier.create_train_test()
    best_k = knn_classifier.cross_validation(X_train, y_train)
    dists = knn_classifier.compute_distances_no_loops(X_test)
    y_test_pred = knn_classifier.predict_labels(dists, k=best_k)
    y_test_pred = y_test_pred.reshape((-1, 1))
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / X_test.shape[0]
    print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))
