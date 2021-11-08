import numpy as np
import tensorflow as tf
import scipy
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)[y_train ==5]
X_valid = x_test.reshape(x_test.shape[0], -1).astype(np.float32)[y_test ==5]
X_train = X_train/255
X_valid = X_valid/255
n_dims = X_train.shape[1]
n_batch = 256
n_hidden = 100
n_layers = 5
# parameters; TODO: should make these class member variables
N_HIDDEN = n_hidden
N_DIMS = n_dims
N_LAYERS = n_layers
def quantile_loss(y_true, y_pred, alpha):
  # note: all inputs have shape (n_batch, n_dims)
  loss_vector = tf.maximum(
      alpha * (y_true - y_pred),
      (alpha - 1) * (y_true - y_pred)
  )
  return tf.reduce_mean(loss_vector, axis=1)
class MAF_RNN_Multi(keras.layers.Layer):
  def __init__(self, n_dims, n_hidden=20, n_layers = 5):
    super(MAF_RNN_Multi, self).__init__()
    self.n_hidden = n_hidden
    self.lstm_list = []
    for i in range(n_layers):
      self.lstm_list.append(keras.layers.LSTM(
          units=n_hidden,
          return_sequences=True,
          return_state=True,
          name = "lstm"+str(i)
      ))
    self.dense = keras.layers.Dense(2, name='maf_dense')

  def __call__(self, input_y, initial_state=[None for i in range(N_LAYERS)]):
    """Run MAF RNN

    When training, this get fed pairs of y, a of length n_dims.
    When sampling, we invoke this one dim at a time and n_dims=1.

    Parameters:
    input_y: (n_batch, n_dims): A tuple of y_prev
    initial_state: tuple of [(n_batch, n_hidden), (n_batch, n_hidden)]

    Ouputs:
    x: (n_batch, n_dims, 2): means and log-variances
    lstm1_h: (n_batch, n_hidden)
    lstm1_c: (n_batch, n_hidden)
    """
    # transform dims
    input_y = tf.expand_dims(input_y, axis=2) # (n_batch, n_dims, 1)
    lstm_h_list = []
    lstm_c_list = []
    lstm_out, lstm_h, lstm_c = self.lstm_list[0](
        input_y, initial_state=initial_state[0]
      )
    lstm_h_list.append(lstm_h)
    lstm_c_list.append(lstm_c)
    lstm_prev = lstm_out
    for i in range(N_LAYERS-1):
      lstm_out, lstm_h, lstm_c = self.lstm_list[i+1](
        lstm_prev, initial_state=initial_state[i+1]
      )
      lstm_h_list.append(lstm_h)
      lstm_c_list.append(lstm_c)

      lstm_out = lstm_out + lstm_prev
      lstm_prev = lstm_out

    # run dense neural net (weights are shared across time steps)
    x = self.dense(lstm_out) # (n_batch, n_dims, 2)

    return x, lstm_h_list, lstm_c_list
# define model
class MAFFlow3(keras.Model):
  def __init__(self, train_type='gaussian', **kwargs):
    super(MAFFlow3, self).__init__(**kwargs)
    self.train_type = train_type
    self.maf_rnn = MAF_RNN_Multi(n_dims=N_DIMS, n_hidden=N_HIDDEN, n_layers = N_LAYERS)
    self.sampling_model = self.create_sampling_model()

    self.loss_tracker = keras.metrics.Mean(name="loss")
    self.llik_tracker = keras.metrics.Mean(name="llik")
    self.quantile_loss_tracker_test = keras.metrics.Mean(name="qlosstest")
    self.llik_tracker_test = keras.metrics.Mean(name="lliktest")
    self.quantile_loss_tracker = keras.metrics.Mean(name="qloss")
    self.llik_test_array = []
    self.qloss_test_array = []

  def create_sampling_model(self):
    # this is used for sampling
    input_y = keras.layers.Input([1]) # input len=1 (for sampling)
    input_h = [keras.layers.Input([N_HIDDEN]) for i in range(N_LAYERS)]
    input_c = [keras.layers.Input([N_HIDDEN]) for i in range(N_LAYERS)]

    # quantile regression super-block
    q_block_out, q_block_h, q_block_c = self.maf_rnn(
        input_y, initial_state=[[input_h[i], input_c[i]] for i in range(N_LAYERS)]
    )

    # create model
    sampling_inputs = [input_y, input_h, input_c]
    sampling_outputs = [q_block_out, q_block_h, q_block_c]
    sampling_model = keras.Model(sampling_inputs, sampling_outputs)

    return sampling_model

  def call(self, inputs):
    input_y, input_a = inputs

    # run quantile block (initial_state=None defaults to zeros)
    q_block_out, _, _ = self.maf_rnn(input_y, initial_state=[None for i in range(N_LAYERS)])
    # ^ (n_batch, n_dims*2)

    # NOTE: ASSUMES A (ALPHA) IS UNIT GAUSSIAN!
    return self.sample_from_gaussian(q_block_out, input_a, self.train_type)

  @property
  def metrics(self):
    return [self.loss_tracker, self.llik_tracker, self.quantile_loss_tracker]

  def eval_test(self):
    y_target = X_valid[:300]
    alpha_unif = tf.random.uniform(shape=[300,N_DIMS]) # (n_batch, n_dims)
    # generate y inputs
    y_input = tf.roll(y_target, shift=1, axis=1)
    y_input = tf.concat(
        [tf.zeros([300, 1]), y_input[:,1:]],
        axis=1
    ) # (n_batch, n_dims)
    with tf.GradientTape() as tape:
      # compute log-likelihood loss
      q_out, _, _ = self.maf_rnn(y_input, initial_state=[None for i in range(N_LAYERS)]) #(n_batch,2*n_dim)
      mu, logsigma = self.rnn_output_to_gaussian(q_out)
      alpha = (y_target - mu) * tf.exp(-logsigma)
      alpha_logprob = -0.5*alpha**2 - np.log(np.sqrt(2.*np.pi))
      alpha_logprob = tf.reduce_sum(alpha_logprob, axis=1) # (n_batch,)
      log_det_jac = tf.reduce_sum(-logsigma, axis=1) # (n_batch,)
      llik_loss = -1.*tf.reduce_mean(alpha_logprob + log_det_jac, axis=0)
      q_pred = self.sample_from_gaussian(q_out, alpha_unif, 'quantile', 'tf')
      q_loss = quantile_loss(y_target, q_pred, alpha_unif)
    return q_loss, llik_loss
  def train_step(self, data):
    # unpack data
    y_target, _ = data
    # n_batch = y_target.shape[0]

    # generate alphas
    alpha_unif = tf.random.uniform(shape=[n_batch,N_DIMS]) # (n_batch, n_dims)
    # generate y inputs
    y_input = tf.roll(y_target, shift=1, axis=1)
    y_input = tf.concat(
        [tf.zeros([n_batch, 1]), y_input[:,1:]],
        axis=1
    ) # (n_batch, n_dims)
    with tf.GradientTape() as tape:
      # compute log-likelihood loss
      q_out, _, _= self.maf_rnn(y_input, initial_state=[None]*N_LAYERS) #(n_batch,2*n_dim)
      mu, logsigma = self.rnn_output_to_gaussian(q_out)

      alpha = (y_target - mu) * tf.exp(-logsigma)
      alpha_logprob = -0.5*(alpha**2) - np.log(np.sqrt(2.*np.pi))
      alpha_logprob = tf.reduce_sum(alpha_logprob, axis=1) # (n_batch,)
      log_det_jac = tf.reduce_sum(-logsigma, axis=1) # (n_batch,)
      llik_loss = -1.*tf.reduce_mean(alpha_logprob + log_det_jac, axis=0)
      q_pred = self.sample_from_gaussian(q_out, alpha_unif, 'quantile', 'tf')
      q_loss = quantile_loss(y_target, q_pred, alpha_unif)
      if self.train_type=='quantile':
        loss = q_loss
      elif self.train_type=='gaussian':
        loss = llik_loss
      else:
        raise ValueError()
    #q_loss_test, llik_loss_test = self.eval_test()
    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    alpha = (y_target - mu) * tf.exp(-logsigma)
    alpha_logprob = -0.5*(alpha**2) - np.log(np.sqrt(2.*np.pi))
    alpha_logprob = tf.reduce_sum(alpha_logprob, axis=1) # (n_batch,)
    log_det_jac = tf.reduce_sum(-logsigma, axis=1) # (n_batch,)
    llik_loss = -1.*tf.reduce_mean(alpha_logprob + log_det_jac, axis=0)
    self.loss_tracker.update_state(loss)
    self.llik_tracker.update_state(llik_loss)
    self.quantile_loss_tracker.update_state(q_loss)
    #self.quantile_loss_tracker_test.update_state(q_loss_test)
    #self.llik_tracker_test.update_state(llik_loss_test)
    # self.llik_test_array.append(self.llik_tracker_test.result().numpy())
    # self.qloss_test_array.append(self.quantile_loss_tracker_test.result().numpy())

    return {
        "loss": self.loss_tracker.result(),
        "qloss": self.quantile_loss_tracker.result(),
        "ll_loss": self.llik_tracker.result(),
        #"qlosstest": self.quantile_loss_tracker_test.result(),
        #"ll_loss_test": self.llik_tracker_test.result(),
      }

  def sample(self, method, n_batch=100):
    # generate arrays to hold data
    y = np.zeros([n_batch, N_DIMS])
    h = [np.zeros([n_batch, N_HIDDEN])]*N_LAYERS
    c = [np.zeros([n_batch, N_HIDDEN])]*N_LAYERS
    if method=='gaussian':
      alpha = np.random.normal(size=(n_batch,N_DIMS))
    elif method=='quantile' or 'mean':
      alpha = np.random.uniform(size=(n_batch,N_DIMS))
    else:
      raise ValueError()

    # generate first sample for y
    q_sample, h, c = self.sampling_model([y[:,[0]], h, c])
    y_sample = np.clip(self.sample_from_gaussian(q_sample, alpha[:,[0]], method, 'np'), 0, 255)
    y[:,0] = np.array(y_sample).flatten()

    # generate remaining samples
    for i in range(1,N_DIMS):
      q_sample, h, c = self.sampling_model([y[:,[i-1]], h, c])
      y_sample = np.clip(self.sample_from_gaussian(q_sample, alpha[:,[i]], method, 'np'), 0, 255)
      y[:,i] = np.array(y_sample).flatten()

    return y

  def rnn_output_to_gaussian(self, maf_rnn_output):
    """NOTE: GAUSSIAN ENCODES p(x|z)"""
    mu = maf_rnn_output[:,:,0] # (n_batch, n_dim)
    logsigma = maf_rnn_output[:,:,1] # (n_batch, n_dim)
    return mu, logsigma

  def sample_from_gaussian(self, maf_rnn_output, alphas, method, mode='tf'):
    mu, logsigma = self.rnn_output_to_gaussian(maf_rnn_output)
    if method=='gaussian' and mode=='tf':
      y_pred = mu + tf.exp(logsigma)*alphas
    elif method=='gaussian' and mode=='np':
      y_pred = mu + np.exp(logsigma)*alphas
    elif method=='quantile' and mode=='tf':
      alphas = tf.clip_by_value(alphas, 1e-2, 1-1e-2)
      y_pred = mu + tf.sqrt(2.)*tf.exp(logsigma)*tf.math.erfinv(2.*alphas-1.)
    elif method=='quantile' and mode=='np':
      alphas = tf.clip_by_value(alphas, 1e-2, 1-1e-2)
      y_pred = mu + np.sqrt(2.)*np.exp(logsigma)*scipy.special.erfinv(2.*alphas-1.)
    elif method=='mean':
      y_pred = mu
    else:
      raise ValueError()

    return y_pred
quantile_regressor = MAFFlow3('gaussian')
quantile_regressor.predict(
    [
     np.ones((10, n_dims)), # y
     np.ones((10, n_dims)), # alpha
    ]
)

checkpoint_filepath = './checkpoints/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='max',
    save_best_only=False)
quantile_regressor.compile(optimizer=keras.optimizers.Adam(1e-3))

# this is a hack because we need to have every batch be n_batch
# correct way to do this is via a tf Dataset generator
n_hack_size = X_train.shape[0] - (X_train.shape[0] % n_batch)
quantile_regressor.fit(X_train[:n_hack_size], np.zeros([n_hack_size,1]), epochs=500, batch_size=n_batch, callbacks=[model_checkpoint_callback])
