import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # compute the loss and the gradient
  num_train = X.shape[0]
  for i in xrange(num_train):
    sumation = 0
    scores = X[i].dot(W)
    largest_element = scores.max()
    scores = scores - largest_element
    correct_class_score = scores[y[i]]
    sumation_values = np.exp(scores)
    sumation = sumation_values.sum()
    loss -= np.log(np.exp(correct_class_score) / sumation)
    p = sumation_values / sumation
    p[y[i]] -= 1
    dW += np.matrix(X[i]).T.dot(np.matrix(p))

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #formula for loss -log[(e^{f_{y_i}}) / (\sum_j e^{f_j})]
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    # compute the loss and the gradient
  num_train = X.shape[0]
  """
  for i in xrange(num_train):
    scores = X[i].dot(W)
    largest_element = scores.max()
    scores = scores - largest_element
    correct_class_score = scores[y[i]]
    sumation_values = np.exp(scores)
    sumation = sumation_values.sum()
    loss -= np.log(np.exp(correct_class_score) / sumation)
    p = sumation_values / sumation
    p[y[i]] -= 1
    dW += np.matrix(X[i]).T.dot(np.matrix(p))
  """
    
  scores = X.dot(W) # shape (N, C)
  largest_elements = scores.max(axis=1) # shape (N, )
  scores -= np.matrix(largest_elements).T
  sumation_values = np.exp(scores) # shape (N, C)
  correct_class_sumation_values = sumation_values[range(num_train), y] # shape (N, )
  sumations = sumation_values.sum(axis=1) # shape (N, )
  loss -= np.sum(np.log(correct_class_sumation_values / sumations))
  p = sumation_values / np.matrix(sumations).T # shape (N, C) / (N, ) = (N, C)
  p[range(num_train), y] -= 1
  dW += X.T.dot(p) # (N, D).T * (N, C) = (D, N) * (N, C) = (D, C)

  """
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  """

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #formula for loss -log[(e^{f_{y_i}}) / (\sum_j e^{f_j})]
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

