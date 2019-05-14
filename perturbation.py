import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import *

# generate noise pattern (shape (1,784))
def fixed_pattern(sigma, probability):
  pattern = []
  p = np.array([probability[0], probability[1], probability[2]])
  size = np.uint16(784)
  for i in range(0,size):
    e = np.random.choice([sigma[0], sigma[1], sigma[2]], p=p.ravel())
    pattern.append(e)
  pattern = np.array(pattern)
  return pattern
  
def generator(inputs, reuse=False):
  with tf.variable_scope("generator", reuse=reuse) as scope:
    x = conv2d(inputs, 32,5,5,1,1, name='conv1')
    x = conv2d(x, 16,1,1,1,1, name='bottleconv')
    x = conv2d(x, 16,3,3,1,1, name='conv2')
    x = conv2d(x, 32,1,1,1,1, name='conv3')
    x = slim.flatten(x, scope='flatten')
    xl = slim.fully_connected(x, 784, activation_fn=None, scope='fc')
    xl = tf.reshape(xl,[-1,28,28,1])
    # xl = linear(x, 1, scope='fc')
    xl = inputs + xl
    xp = tf.nn.tanh(xl)
    # xl = tf.reshape(xl,[-1,28,28,1])
    return xl, xp