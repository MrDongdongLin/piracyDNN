import tensorflow as tf
import tensorflow.contrib.slim as slim
from cleverhans.model import Model

def conv2d(input_,
           output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2,
           stddev=0.01, 
           name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
      initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
  return conv
  
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.random_normal_initializer(stddev=stddev))
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    if with_w:
      return deconv, w, biases
    else:
      return deconv
    
def linear(input_,
           output_size,
           scope=None,
           stddev=0.02,
           bias_start=0.0,
           with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], 
      tf.float32, tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
    initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def res_identity(input_tensor,conv_depth,kernel_shape,layer_name):
  with tf.variable_scope(layer_name):
    relu = tf.nn.relu(slim.conv2d(input_tensor,conv_depth,kernel_shape))
    outputs = tf.nn.relu(slim.conv2d(relu,conv_depth,kernel_shape) + input_tensor)
  return outputs

def res_change(input_tensor,conv_depth,kernel_shape,layer_name):
  with tf.variable_scope(layer_name):
    relu = tf.nn.relu(slim.conv2d(input_tensor,conv_depth,kernel_shape,stride=2))
    input_tensor_reshape = slim.conv2d(input_tensor,conv_depth,[1,1],stride=2)
    outputs = tf.nn.relu(slim.conv2d(relu,conv_depth,kernel_shape) + input_tensor_reshape)
  return outputs

class MyModel(object):
  def __init__(self, num_classes):
    self.num_classes = num_classes
  
  def basic_cnn(self, inputs, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
      x = conv2d(inputs, 32,3,3,1,1, name='conv1')
      # x = slim.conv2d(inputs, 32, [3,3], scope='conv1')
      x = tf.nn.relu(x)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      x = conv2d(x, 64,3,3,1,1, name='conv2')
      x = tf.nn.relu(x)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      x = tf.reshape(x, [-1, 7 * 7 * 64])
      x = linear(x,1024,scope='fc1')
      x = tf.nn.relu(x)
      x = tf.nn.dropout(x, 0.5)
      logist = linear(x,self.num_classes,scope='fc2')
      logist_softmax = tf.nn.softmax(logist, name="softmax")
    return logist, logist_softmax
  
  def resnet20(self, inputs, reuse=False):
    with tf.variable_scope("resnet", reuse=reuse) as scope:
      x = tf.reshape(inputs,[-1,28,28,1])
      conv_1 = tf.nn.relu(slim.conv2d(x,32,[3,3])) #28 * 28 * 32
      pool_1 = slim.max_pool2d(conv_1,[2,2]) # 14 * 14 * 32
      block_1 = res_identity(pool_1,32,[3,3],'layer_2')
      block_2 = res_change(block_1,64,[3,3],'layer_3')
      block_3 = res_identity(block_2,64,[3,3],'layer_4')
      block_4 = res_change(block_3,32,[3,3],'layer_5')
      net_flatten = slim.flatten(block_4,scope='flatten')
      fc_1 = slim.fully_connected(slim.dropout(net_flatten,0.8),200,activation_fn=tf.nn.tanh,scope='fc_1')
      output = slim.fully_connected(slim.dropout(fc_1,0.8),10,activation_fn=None,scope='fc2')
      logist_softmax = tf.nn.softmax(output, name="softmax")
    return output, logist_softmax

class CNN(Model):
  def __init__(self, scope, nb_classes):
    super(CNN, self).__init__(nb_classes=nb_classes, needs_dummy_fprop=True)
    self.nb_classes = nb_classes
  
  def fprop(self, inputs):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:
      x = conv2d(inputs, 32,3,3,1,1, name='conv1')
      # x = slim.conv2d(inputs, 32, [3,3], scope='conv1')
      x = tf.nn.relu(x)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      x = conv2d(x, 64,3,3,1,1, name='conv2')
      x = tf.nn.relu(x)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      x = tf.reshape(x, [-1, 7 * 7 * 64])
      x = linear(x,1024,scope='fc1')
      x = tf.nn.relu(x)
      x = tf.nn.dropout(x, 0.5)
      logist = linear(x,self.nb_classes,scope='fc2')
      logist_softmax = tf.nn.softmax(logist, name="softmax")
    return {self.O_LOGITS:logist, self.O_PROBS:logist_softmax}