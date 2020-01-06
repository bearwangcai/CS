from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


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
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                train_cor = self.X_train[j,:] #[D,]
                #print('train_cor', train_cor.shape)
                test_cor = X[i,:] #[D,]
                #print('test_cor', test_cor.shape)
                train_square = np.dot(train_cor,train_cor)
                test_square = np.dot(test_cor,test_cor)
                train_test = np.dot(train_cor,test_cor)
                dists[i,j] = np.sqrt(train_square + test_square - 2 * train_test)
                
                dist_verify = np.linalg.norm(train_cor - test_cor)
                if dists[i, j] == dist_verify:
                    pass
                else:
                    print('error')


                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            test_square = X[i,:].dot(X[i,:]) #(1,1)
            
            train_test = self.X_train.dot(X[i,:].T)  # (num_train,1)

            train_dot = self.X_train.dot(self.X_train.T) #(num_train,num_train)            
            train_square_pre = train_dot * np.eye(num_train)  # (num_train,num_train)
                
            train_square = np.sum(train_square_pre, axis=1)  # (num_train,1)
            
            dists[i,:] = np.sqrt(train_square + test_square - 2 * train_test).T #(1, num_train)


            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

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
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        test_dot = X.dot(X.T)  # (num_test,num_test)
        test_square_pre = test_dot * np.eye(num_test)  # (num_test,num_test)
        test_square = np.sum(test_square_pre, axis=1).reshape(-1,1) #(num_test,1)

        test_train = X.dot(self.X_train.T)  # (num_test,num_train)

        train_dot = self.X_train.dot(self.X_train.T)  # (num_train,num_train)
        train_square_pre = train_dot * np.eye(num_train)  # (num_train,num_train)
        train_square = np.sum(train_square_pre, axis=0).reshape(1,-1)  # (1,num_train)
        
        dists= np.sqrt(test_square - 2 * test_train + train_square)  # (num_test, num_train)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

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
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #print(i)
            y_sort = np.argsort(dists[i, :])
            '''
            for j in range(k):
                print("k", k)
                print("j", j)
                print("y_sort[j]", y_sort[j])
                print("self.y_train[y_sort[j]]", self.y_train[y_sort[j]])
                closest_y.append(self.y_train[y_sort[j]])
            '''
            #print("y_sort", y_sort)
            #print("self.y_train", self.y_train.shape)
            #print(k)
            closest_y = [int(self.y_train[y_sort[j]]) for j in range(k)]
            #print(closest_y)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            closest_y_set = sorted(list(set(closest_y)))
            
            closest_y_dict = {}
            for o in closest_y_set:
                closest_y_dict[o] = 0
            for m in closest_y:
                closest_y_dict[m] += 1

            #print(closest_y_set)
            count = np.array([closest_y_dict[n] for n in closest_y_set])
            y_pred[i] = closest_y_set[np.argmax(count)]

            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
