# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:11:08 2016

@author: rob
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops


class VAE_fc():
  def __init__(self,config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_enc1 = config['num_enc1']
    num_enc2 = config['num_enc2']
    num_l = config['num_l']
    D = config['D']

    #Function for initialization
    def xv_init(arg_in, arg_out,shape=None):
      low = -np.sqrt(6.0/(arg_in + arg_out))
      high = np.sqrt(6.0/(arg_in + arg_out))
      if shape is None:
        tensor_shape = (arg_in, arg_out)
      return tf.random_uniform(tensor_shape, minval=low, maxval=high, dtype=tf.float32)

    with tf.name_scope("Placeholders") as scope:
      self.x = tf.placeholder("float", shape=[None, D], name = 'Input_data')

    with tf.name_scope("Encoding_network") as scope:
      #Layer 1
      W1e = tf.Variable(xv_init(D,num_enc1))
      b1e = tf.Variable(tf.constant(0.1,shape=[num_enc1],dtype=tf.float32))
      h1e = tf.nn.relu(tf.nn.xw_plus_b(self.x,W1e,b1e))

      #Layer 1
      W2e = tf.Variable(xv_init(num_enc1,num_enc2))
      b2e = tf.Variable(tf.constant(0.1,shape=[num_enc2],dtype=tf.float32))
      h2e = tf.nn.relu(tf.nn.xw_plus_b(h1e,W2e,b2e))

      #layer for mean of z
      W_mu = tf.Variable(xv_init(num_enc2,num_l))
      b_mu = tf.Variable(tf.constant(0.1,shape=[num_l],dtype=tf.float32))
      self.z_mu = tf.nn.xw_plus_b(h2e,W_mu,b_mu)  #mu, mean, of latent space

      #layer for sigma of z
      W_sig = tf.Variable(xv_init(num_enc2,num_l))
      b_sig = tf.Variable(tf.constant(0.1,shape=[num_l],dtype=tf.float32))
      z_sig_log_sq = tf.nn.xw_plus_b(h2e,W_sig,b_sig)  #sigma of latent space, in log-scale and squared.
      # This log_sq will save computation later on. log(sig^2) is a real number, so no sigmoid is necessary

    with tf.name_scope("Latent_space") as scope:
      eps = tf.random_normal(tf.shape(self.z_mu),0,1,dtype=tf.float32)
      self.z = self.z_mu + tf.mul(tf.sqrt(tf.exp(z_sig_log_sq)),eps)

    with tf.name_scope("Decoding_network") as scope:
      #Layer 1
      W1d = tf.Variable(xv_init(num_l,num_enc2))
      b1d = tf.Variable(tf.constant(0.1,shape=[num_enc2],dtype=tf.float32))
      h1d = tf.nn.relu(tf.nn.xw_plus_b(self.z,W1d,b1d))

      #Layer 1
      W2d = tf.Variable(xv_init(num_enc2,num_enc1))
      b2d = tf.Variable(tf.constant(0.01,shape=[num_enc1],dtype=tf.float32))
      h2d = tf.nn.relu(tf.nn.xw_plus_b(h1d,W2d,b2d))

      #Layer for reconstruction
      W_rec = tf.Variable(xv_init(num_enc1,D))
      b_rec = tf.Variable(tf.constant(0.5,shape=[D],dtype=tf.float32))
      self.rec = tf.nn.sigmoid(tf.nn.xw_plus_b(h2d,W_rec,b_rec))  #Reconstruction. FOr now only mean

    with tf.name_scope("Loss_calculation") as scope:
      #See equation (10) of https://arxiv.org/abs/1312.6114
      loss_rec = tf.reduce_sum(self.x * tf.log(1e-10 + self.rec) + (1-self.x) * tf.log(1-self.rec+1e-10),1)  #Add 1e-10 to avoid numeric instability
      loss_kld = 0.5*tf.reduce_sum((1+z_sig_log_sq-tf.square(self.z_mu)-tf.exp(z_sig_log_sq)),1)   #KL divergence

      self.cost = -1*tf.reduce_mean(loss_rec + loss_kld)

    with tf.name_scope("Optimization") as scope:
      tvars = tf.trainable_variables()
      #We clip the gradients to prevent explosion
      grads = tf.gradients(self.cost, tvars)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients = zip(grads, tvars)
      self.train_step = optimizer.apply_gradients(gradients)
      # The following block plots for every trainable variable
      #  - Histogram of the entries of the Tensor
      #  - Histogram of the gradient over the Tensor
      #  - Histogram of the grradient-norm over the Tensor
      numel = tf.constant([[0]])
      for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
          grad_values = gradient.values
        else:
          grad_values = gradient

        numel +=tf.reduce_sum(tf.size(variable))
        h1 = tf.histogram_summary(variable.name, variable)
        h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
        h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

    #Define one op to call all summaries
    self.merged = tf.merge_all_summaries()
    print('Finished computation graph')


