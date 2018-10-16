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
  num_X = X.shape[0]
  num_class = W.shape[1]
  for i in range(0,num_X):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        p = np.exp(scores[y[i]])/np.sum(np.exp(scores))
        loss += -np.log(p)
        for j in range(0,num_class):
            dW[:,j] += X[i].T * (np.exp(scores[j])/np.sum(np.exp(scores)))
            if j == y[i]:
                dW[:,j] -= X[i].T
  loss /= num_X         
  loss += reg*np.sum(W*W)
  dW /= num_X
  dW += 2*reg*W
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
  num_X = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  denom = np.sum(exp_scores,axis=1, keepdims=True)
  probs = exp_scores/denom
  loss = np.sum(-np.log(probs[np.arange(num_X),y]))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_X
  loss += reg*np.sum(W*W)
    
  ind = np.zeros_like(probs)
  ind[np.arange(num_X), y] = 1
  dW = X.T.dot(probs - ind)
  dW /= num_X
  dW += reg*W
    
  return loss, dW

