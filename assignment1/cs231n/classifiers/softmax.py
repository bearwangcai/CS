from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    dW_ = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = np.zeros((1,num_classes))
    #scores_correct = []

    for i_num_train in range(num_train):
      x_train = X[i_num_train,:].reshape(1,-1) #(1,D)
      score_correct = 0
      for i_num_class in range(num_classes):
        w_each_class = W[:,i_num_class].reshape(-1,1) #(D,1)
        score = x_train.dot(w_each_class)
        dW_[:, i_num_class] = x_train.flatten()
        scores[0,i_num_class] = score
        if (i_num_class == y[i_num_train]):
          score_correct = score
          #scores_correct.append(score_correct)

      score_max = np.max(scores)
      class_max = np.argmax(scores)
      scores -= score_max
      score_exp = np.exp(scores)
      score_correct_exp = np.exp(score_correct - score_max)
      score_sum = np.sum(score_exp)
      dW_ = (-score_correct_exp / score_sum**2) * scores * dW_
      dW_[:, y[i_num_train]] = score_correct_exp * \
          (score_sum - score_correct_exp) / score_sum**2 * x_train
      dW_[:, class_max] = - score_correct_exp / \
          score_sum**2 * x_train


      loss -= np.log(score_correct_exp / score_sum)
      #print(loss)
      dW -= 1 / (score_correct_exp / score_sum) * dW_ 


    loss /= num_train
    dW /=num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
    #return loss, dW, scores_correct


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.ones_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W) #(N,C)
    num_train = X.shape[0]
    #print("X, axis=0", np.sum(X, axis=0).shape) #(D,)
    

    scores_max = np.max(scores, axis=1).reshape(-1,1)
    #print("scores_max", scores_max.shape) #(N,)
    scores_max_arg = np.argmax(scores, axis=1)

    correct = np.zeros_like(scores)
    correct[range(num_train), y] = 1
    #print("correct", correct)
    scores -= scores_max
    scores_exp = np.exp(scores) #(N,C)
    #print("scores_exp", scores_exp)
    #print("scores_correct", scores_correct)
    scores_correct = scores_exp * correct  # (N,C)
    #print("scores_correct", scores_correct)
    scores_correct = np.sum(scores_correct, axis=1).reshape(-1, 1)  # (N,1)
    #print("scores_correct", scores_correct.shape)
    #scores_correct_exp = np.exp(scores_correct) #(N,1)
    scores_correct_exp = scores_correct
    #print(scores_max)
    scores_sum = np.sum(scores_exp, axis=1).reshape(-1,1) #(N,1)
    #print("scores_sum",scores_sum)

    loss = -np.sum(np.log(scores_correct_exp / scores_sum))

    #print("(scores_sum - scores_correct_exp)", (scores_sum - scores_correct_exp).shape)
    scores[range(num_train), y] =  -(scores_sum - scores_correct_exp).flatten() 
    scores[range(num_train), scores_max_arg] =  1
    dW = X.T.dot((-scores_correct_exp / scores_sum**2) * scores / -(scores_correct_exp / scores_sum))


    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
