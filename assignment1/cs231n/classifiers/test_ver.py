from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange


def loss(X, std, input_size, hidden_size, output_size, y=None, reg=0.0 ):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
        an integer in the range 0 <= y[i] < C. This parameter is optional; if it
        is not passed then we only return scores, and if it is passed then we
        instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
        samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
        with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = std * np.random.randn(input_size,
                                   hidden_size), np.zeros(hidden_size)
    W2, b2 = std * np.random.randn(hidden_size,
                                   output_size), np.zeros(output_size)
    N, D = X.shape
    H, C = W2.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    a1 = X.dot(W1) + b1 # (N, H)
    dW1 = (X.T).dot(np.ones_like(a1))  # (D,H)
    db1 = (np.ones((N,1)).T).dot(np.ones_like(a1)) #(1,H)

    cache = np.ones((N,H))
    cache[a1 < 0] = 0 #(N,H)
    a1[a1 < 0] = 0 #(N,H)
    da1_cache = cache * np.ones_like(a1) #(N,H)
    dW1 = X.T.dot(da1_cache)  # (D,H)
    db1 = (np.ones((N,1)).T).dot(da1_cache) #(1,H)

    scores = a1.dot(W2) + b2 #(N,C)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW2 = a1.T.dot(np.ones_like(scores)) #(H,C)
    db2 = (np.ones((N,1)).T).dot(np.ones_like(scores)) #(1,C)
    #da1 = (W2.dot(np.ones((C,N)))).T #(N,H)
    da1 = np.ones_like(scores).dot(W2.T) #(N,H)
    dW1 = X.T.dot(da1 * da1_cache) #(D,H)
    db1 = (np.ones((N,1)).T).dot(da1 * da1_cache) #(1,H)

    scores_max = np.max(scores, axis=1) #(N,)
    scores_max_arg = np.argmax(scores, axis=1) #(N,)
    
    scores -= scores_max.reshape(-1,1)  # (N,C)
    dscores_max_pre = np.zeros_like(scores) #(N,C)
    dscores_max_pre[range(N),scores_max_arg] = 1  #(N,C)

    dW2 = a1.T.dot(np.ones_like(scores)) - a1.T.dot(dscores_max_pre)  # (H,C)
    db2 = (np.ones((N, 1)).T).dot(np.ones_like(scores)
                                    ) - (np.ones((N, 1)).T).dot(dscores_max_pre)  # (1,C)
    da1 = np.ones_like(scores).dot(W2.T) - (np.ones_like(scores).dot(np.ones((C,1)))) * dscores_max_pre.dot(W2.T)  # (N,H)
    dW1 = X.T.dot(da1 * da1_cache)  # (D,H)
    db1 = (np.ones((N,1)).T).dot(da1 * da1_cache) #(1,H)

    scores = np.exp(scores)  # (N,C)
    dW2 = a1.T.dot(scores * np.ones_like(scores)) - \
        a1.T.dot(scores * dscores_max_pre)  # (H,C)
    db2 = (np.ones((N, 1)).T).dot(scores * np.ones_like(scores)
                                    ) - (np.ones((N, 1)).T).dot(scores * dscores_max_pre)  # (1,C)
    da1 = (scores * np.ones_like(scores)).dot(W2.T) - \
        (scores * np.ones_like(scores)).dot(np.ones((C, 1))) * \
        dscores_max_pre.dot(W2.T)  # (N,H)
    dW1 = X.T.dot(da1 * da1_cache)  # (D,H)
    db1 = (np.ones((N, 1)).T).dot(da1 * da1_cache)  # (1,H)


    scores_correct = np.zeros_like(scores)
    scores_correct[range(N),y] = 1
    scores_correct =  scores_correct * scores

    loss = -np.log(np.sum(scores_correct, axis=1) /
                    np.sum(scores, axis=1))  # (N,)
    
    scores_sum = np.sum(scores, axis=1).reshape(N,1)
    scores_correct_exp = (np.sum(scores_correct, axis=1)).reshape(N, 1)
    '''
    scores[range(N), y] = -(scores_sum - scores_correct_exp).flatten()
    scores[range(N), scores_max_arg] = 1
    dW2 = a1.T.dot((-scores_correct_exp / scores_sum**2) *
                scores / -(scores_correct_exp / scores_sum))
    '''
    dW21 = a1.T.dot(scores_correct * (scores - scores * dscores_max_pre) + scores * scores_correct)  # (H,C)
    
    
    scores[range(N), y] = -(scores_sum - scores_correct_exp).flatten()
    scores[range(N), scores_max_arg] = 1
    dW22 = a1.T.dot((-scores_correct_exp / scores_sum**2) *
                scores / -(scores_correct_exp / scores_sum))
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return dW22-dW21

std = 1e-4
input_size, hidden_size, output_size = 3,5,4
X = np.random.randn(5, 3)
y = np.array([1,3,2]).reshape(3,1)
b = loss(X, std, input_size, hidden_size, output_size, y)
c = np.sum(b)
print(b)
