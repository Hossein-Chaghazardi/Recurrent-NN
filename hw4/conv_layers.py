import numpy as np
from layers import batchnorm_forward, batchnorm_backward

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C1, H, W = x.shape
  F, C2, HH, WW = w.shape
  assert (C1 == C2)
  C = C1
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N, F, H_out, W_out))  # Pre allocation
  x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
  w_matricized = np.reshape(w, (F, -1)).T
  pointer_width, pointer_height = 0, 0
  for w_idx in xrange(W_out):
    pointer_height = 0
    for h_idx in xrange(H_out):
      x_sub_selection = x_padded[:, :, pointer_height:pointer_height+HH, pointer_width:pointer_width+WW].reshape(N, -1)
      out[:, :, h_idx, w_idx] = x_sub_selection.dot(w_matricized) + b  # shape: (N, F)
      pointer_height += stride
    pointer_width += stride

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C1, H, W = x.shape
  F, C2, HH, WW = w.shape
  assert (C1 == C2)
  C = C1

  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride

  # Padding and reshaping
  x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
  w_matricized = np.reshape(w, (F, -1)).T  # shape: (F, -1).T

  # Pre-allocation
  dx = np.zeros_like(x_padded)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  dw_reshaped = np.reshape(dw, (F, -1)).T


  # We have to iterate for dx
  pointer_width, pointer_height = 0, 0
  for w_idx in xrange(W_out):
    pointer_height = 0
    for h_idx in xrange(H_out):
      # Calculating dx
      dout_sub = dout[:, :, h_idx, w_idx]  # shape: (N, F)
      dx_this_pixel = dout_sub.dot(w_matricized.T)  # (N, C*HH*WW)
      # Updating the part of dx which was affected by dout at H_out  W_out
      dx[:, :, pointer_height:pointer_height+HH, pointer_width:pointer_width+WW] += dx_this_pixel.reshape(N, C, HH, WW)
      # Updating db which is derived directly from dout
      x_sub_reshaped = x_padded[:, :, pointer_height:pointer_height+HH, pointer_width:pointer_width+WW].reshape(N, -1)  #shape:(N, C*WW*HH)
      dw_reshaped += x_sub_reshaped.T.dot(dout[:, :, h_idx, w_idx])
      db += np.sum(dout[:, :, h_idx, w_idx], axis=0)

      # Adding counters for sweeping
      pointer_height += stride
    pointer_width += stride

  dx = dx[:, :, pad:-pad, pad:-pad]  # This is very tricky, we have to get rid of the padding in the backpropagation
  dw = np.reshape(dw_reshaped.T, w.shape)  # We transpose before reshaping to keep track of the order of dimesnions
  db = np.reshape(db, b.shape)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param, switches)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # Extracting params
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  pad = 0  # For pooling, there is no padding
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride

  # Iterating
  pointer_height, pointer_width, out = 0, 0, np.zeros((N, C, H_out, W_out))
  for heigh_idx in xrange(H_out):
    pointer_width = 0
    for width_idx in xrange(W_out):
      out[:, :, heigh_idx, width_idx] = x[:, :, pointer_height:pointer_height+HH, pointer_width:pointer_width+WW].max((2,3))
      pointer_width += stride
    pointer_height += stride

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param, switches) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # Extracting Params
  x, pool_param = cache
  N, C, H, W = x.shape
  N, C, H_out, W_out = dout.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  # Pre allocation
  dx = np.zeros_like(x)

  # Iterate through dout
  pointer_height = 0
  for heigh_idx in xrange(H_out):
    pointer_width = 0
    for width_idx in xrange(W_out):
      d_out_reshaped = dout[:, :, heigh_idx, width_idx].reshape(N*C)  # Shape: (N*C)
      # Finding the maximum in x.reshaped along 1
      max_args = x[:, :, pointer_height:pointer_height+pool_height, pointer_width:pointer_width+pool_width].reshape(N*C, -1).argmax(axis=1)
      # Inserting the numbers in dx
      dx_reshaped = np.zeros((N*C, pool_height*pool_height))
      dx_reshaped[range(N*C), max_args] = d_out_reshaped
      dx[:, :, pointer_height:pointer_height+pool_height, pointer_width:pointer_width+pool_width] = dx_reshaped.reshape(N , C, pool_height, pool_width)
      # Adding the strides
      pointer_width += stride
    pointer_height += stride

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (C,) giving running mean of features
    - running_var Array of shape (C,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined in hw2. Your implementation should #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape
  x_prime = x.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
  out_prime, cache = batchnorm_forward(x=x_prime, gamma=gamma, beta=beta, bn_param=bn_param)
  out = out_prime.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined in hw2. Your implementation should #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape
  dout_prime = dout.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
  dx_prime, dgamma, dbeta = batchnorm_backward(dout_prime, cache)
  dx = dx_prime.reshape(N, H, W, C).transpose(0, 3, 1, 2)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  
