import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
n_residual_blocks = 5
n_batch = 128
# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x = x_train[y_train ==5]
x_valid = x_test[y_test ==5]
y =y_train[y_train==5]
y_valid = y_test[y_test==5]
# Concatenate all of the images together
data = np.concatenate((x, x_valid), axis=0)
data = np.expand_dims(data, axis=-1)
# Round all pixel values less than 33% of the max 256 value to 0
# anything above this value gets rounded up to 1 so that all values are either
# 0 or 1
#data = np.where(data < (0.33 * 256), 0, 1)
data = data/255
data = data.astype(np.float32)
#keep in mind dimensions
def quantile_loss(y_true, y_pred, alpha):
  # note: all inputs have shape (n_batch, n_dims)
  loss_vector = tf.maximum(
      alpha * (y_true - y_pred),
      (alpha - 1) * (y_true - y_pred)
  )
  return tf.reduce_mean(loss_vector, axis=[1,2,3])

# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])
input = keras.Input(input_shape)
input_layer = tf.keras.layers.InputLayer(input_shape=(28, 28, 1))
pixel_conv1 = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
)
residual_blocks = []
for _ in range(n_residual_blocks):
    residual_blocks.append(ResidualBlock(filters=128))
pixel_conv2 = PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )
pixel_conv3 = PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )
conv2d = keras.layers.Dense(2, "dense")
x = input
x = pixel_conv1(x)
for i in range(n_residual_blocks):
    x = residual_blocks[i](x)
x = pixel_conv2(x)
x = pixel_conv3(x)
out = conv2d(x)

pixel_cnn = keras.Model(input, out)
adam = keras.optimizers.Adam(learning_rate=0.0005)

pixel_cnn.summary()
train_dataset = tf.data.Dataset.from_tensor_slices((data, data))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(n_batch)
def output_to_gaussian(maf_rnn_output):
    """NOTE: GAUSSIAN ENCODES p(x|z)"""
    mu = tf.expand_dims(maf_rnn_output[:,:,:,0], 3) # (n_batch, n_dim)
    logsigma = tf.expand_dims(maf_rnn_output[:,:,:,1], 3) # (n_batch, n_dim)
    return mu, logsigma
checkpoint_filepath = './checkpoints_pixelcnn/checkpoint-{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
                save_weights_only=True)
epochs = 400
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,), flush=True))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      # generate alphas
      alpha_unif = tf.random.uniform(shape=[n_batch, 28, 28, 1]) # (n_batch, n_dims)
      with tf.GradientTape() as tape:
        # compute log-likelihood loss
        out = pixel_cnn(x_batch_train)

        mu, logsigma = output_to_gaussian(out)

        alpha = (y_batch_train - mu) * tf.exp(-logsigma)
        alpha_logprob = -0.5*(alpha**2) - np.log(np.sqrt(2.*np.pi))
        alpha_logprob = tf.reduce_sum(alpha_logprob, axis=[1,2,3]) # (n_batch,)
        log_det_jac = tf.reduce_sum(-logsigma, axis=[1,2,3]) # (n_batch,)
        llik_loss = -1.*tf.reduce_mean(alpha_logprob + log_det_jac, axis=0)
        # q_pred = self.sample_from_gaussian(q_out, alpha_unif, 'quantile', 'tf')
        # q_loss = quantile_loss(y_target, q_pred, alpha_unif)
        # if self.train_type=='quantile':
        #   loss = q_loss
        # elif self.train_type=='gaussian':
        loss = llik_loss
        # else:
        #   raise ValueError()
        #q_loss_test, llik_loss_test = self.eval_test()
        trainable_vars = pixel_cnn.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        adam.apply_gradients(zip(grads, trainable_vars))


        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss))
            )
            print("Seen so far: %s samples" % ((step + 1) * 64), flush=True)
    pixel_cnn.save_weights(checkpoint_filepath.format(epoch=epoch))
batch = 1
pixels = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols, channels = pixels.shape
alpha = np.random.normal(size=(batch,28,28))
# Iterate over the pixels because generation has to be done sequentially pixel by pixel.
for row in tqdm(range(rows)):
    for col in range(cols):
        mu, sigma = output_to_gaussian(pixel_cnn.predict(pixels))
        mu_value = mu[:, row, col, 0]
        sigma_value = sigma[:, row, col, 0]
        # Use the probabilities to pick pixel values and append the values to the image
        # frame.
        pixels[:, row, col, 0] = np.clip(mu_value + tf.exp(sigma_value)*alpha[:, row, col], 0, 1)
from IPython.display import Image, display
def deprocess_image(x):
    x *= 255.0
    # Convert to uint8 and clip to the valid range [0, 255]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

# Iterate over the generated images and plot them with matplotlib.
for i, pic in enumerate(pixels):
    keras.preprocessing.image.save_img(
        "generated_image_{}.png".format(i), deprocess_image(pic)
    )

display(Image("generated_image_0.png"))
# class PixelCNN_MAF(keras.Model):
#   def __init__(self, train_type='gaussian', **kwargs):
#       super(PixelCNN_MAF, self).__init__(**kwargs)
#       self.train_type = train_type
#       self.pixel_conv1 = PixelConvLayer(
#           mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
#       )
#       self.residual_blocks = []
#       for _ in range(n_residual_blocks):
#           self.residual_blocks.append(ResidualBlock(filters=128))
#       self.pixel_conv2 = PixelConvLayer(
#               mask_type="B",
#               filters=128,
#               kernel_size=1,
#               strides=1,
#               activation="relu",
#               padding="valid",
#           )
#       self.pixel_conv3 = PixelConvLayer(
#               mask_type="B",
#               filters=128,
#               kernel_size=1,
#               strides=1,
#               activation="relu",
#               padding="valid",
#           )
#
#       self.conv2d = keras.layers.Conv2D(
#           filters=2, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
#       )
#   def output_to_gaussian(self, maf_rnn_output):
#       """NOTE: GAUSSIAN ENCODES p(x|z)"""
#       mu = tf.expand_dims(maf_rnn_output[:,:,:,0], 3) # (n_batch, n_dim)
#       logsigma = tf.expand_dims(maf_rnn_output[:,:,:,1], 3) # (n_batch, n_dim)
#       return mu, logsigma
#   def call(self, inputs):
#       input_y, input_a = inputs
#       # run quantile block (initial_state=None defaults to zeros)
#       x = self.pixel_conv1(input_y)
#       for i in range(n_residual_blocks):
#           x = self.residual_blocks[i](x)
#       x = self.pixel_conv2(x)
#       x = self.pixel_conv3(x)
#       out = self.conv2d(x)
#
#       mu, logsigma = self.output_to_gaussian(out)
#       # ^ (n_batch, n_dims*2)
#
#       # NOTE: ASSUMES A (ALPHA) IS UNIT GAUSSIAN!
#       return self.sample_from_gaussian(out, input_a, self.train_type)
#   def sample_from_gaussian(self, maf_rnn_output, alphas, method, mode='tf'):
#     mu, logsigma = self.output_to_gaussian(maf_rnn_output)
#     if method=='gaussian' and mode=='tf':
#       y_pred = mu + tf.exp(logsigma)*alphas
#     elif method=='gaussian' and mode=='np':
#       y_pred = mu + np.exp(logsigma)*alphas
#     elif method=='quantile' and mode=='tf':
#       alphas = tf.clip_by_value(alphas, 1e-2, 1-1e-2)
#       y_pred = mu + tf.sqrt(2.)*tf.exp(logsigma)*tf.math.erfinv(2.*alphas-1.)
#     elif method=='quantile' and mode=='np':
#       alphas = tf.clip_by_value(alphas, 1e-2, 1-1e-2)
#       y_pred = mu + np.sqrt(2.)*np.exp(logsigma)*scipy.special.erfinv(2.*alphas-1.)
#     elif method=='mean':
#       y_pred = mu
#     else:
#       raise ValueError()
#
#     return y_pred
#   def train_step(self, data):
#     y_target, _ = data
#
#     # generate alphas
#     alpha_unif = tf.random.uniform(shape=[n_batch, 28, 28, 1]) # (n_batch, n_dims)
#     with tf.GradientTape() as tape:
#       # compute log-likelihood loss
#       x = self.pixel_conv1(y_target)
#       for i in range(n_residual_blocks):
#           x = self.residual_blocks[i](x)
#       x = self.pixel_conv2(x)
#       x = self.pixel_conv3(x)
#       out = self.conv2d(x)
#
#       mu, logsigma = self.output_to_gaussian(out)
#
#       alpha = (y_target - mu) * tf.exp(-logsigma)
#       alpha_logprob = -0.5*(alpha**2) - np.log(np.sqrt(2.*np.pi))
#       alpha_logprob = tf.reduce_sum(alpha_logprob, axis=[1,2,3]) # (n_batch,)
#       log_det_jac = tf.reduce_sum(-logsigma, axis=[1,2,3]) # (n_batch,)
#       llik_loss = -1.*tf.reduce_mean(alpha_logprob + log_det_jac, axis=0)
#       # q_pred = self.sample_from_gaussian(q_out, alpha_unif, 'quantile', 'tf')
#       # q_loss = quantile_loss(y_target, q_pred, alpha_unif)
#       # if self.train_type=='quantile':
#       #   loss = q_loss
#       # elif self.train_type=='gaussian':
#       loss = llik_loss
#       # else:
#       #   raise ValueError()
#     #q_loss_test, llik_loss_test = self.eval_test()
#     trainable_vars = self.trainable_variables
#     grads = tape.gradient(loss, trainable_vars)
#     self.optimizer.apply_gradients(zip(grads, trainable_vars))
#     # alpha = (y_target - mu) * tf.exp(-logsigma)
#     # alpha_logprob = -0.5*(alpha**2) - np.log(np.sqrt(2.*np.pi))
#     # alpha_logprob = tf.reduce_sum(alpha_logprob, axis=1) # (n_batch,)
#     # log_det_jac = tf.reduce_sum(-logsigma, axis=1) # (n_batch,)
#     # llik_loss = -1.*tf.reduce_mean(alpha_logprob + log_det_jac, axis=0)
#     # self.loss_tracker.update_state(loss)
#     # self.llik_tracker.update_state(llik_loss)
#     # self.quantile_loss_tracker.update_state(q_loss)
#     #self.quantile_loss_tracker_test.update_state(q_loss_test)
#     #self.llik_tracker_test.update_state(llik_loss_test)
#     # self.llik_test_array.append(self.llik_tracker_test.result().numpy())
#     # self.qloss_test_array.append(self.quantile_loss_tracker_test.result().numpy())
#
#     # return {
#     #     "loss": self.loss_tracker.result(),
#     #     "qloss": self.quantile_loss_tracker.result(),
#     #     "ll_loss": self.llik_tracker.result(),
#     #     #"qlosstest": self.quantile_loss_tracker_test.result(),
#     #     #"ll_loss_test": self.llik_tracker_test.result(),
#     #   }
#     return
#
# pixel_cnn = PixelCNN_MAF("gaussian")
# pixel_cnn.predict(
#     [
#      np.ones((10, 28, 28, 1)), # y
#      np.ones((10, 28, 28, 1)), # alpha
#     ]
# )
# adam = keras.optimizers.Adam(learning_rate=0.0005)
#
# pixel_cnn.compile(optimizer=adam)
#
# pixel_cnn.summary()
#
# checkpoint_filepath = './checkpoints_pixelcnn/checkpoint'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#             filepath=checkpoint_filepath,
#                 save_weights_only=True,
#                     monitor='loss',
#                         mode='max',
#                             save_best_only=False)
# pixel_cnn.fit(
#     x=data, y=np.zeros_like(data), batch_size=128, epochs=399, validation_split=0.1, verbose=2, callbacks=[model_checkpoint_callback]
# )
# from IPython.display import Image, display
#
# # Create an empty array of pixels.
# batch = 4
# pixels = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
# batch, rows, cols, channels = pixels.shape
#
# # Iterate over the pixels because generation has to be done sequentially pixel by pixel.
# for row in tqdm(range(rows)):
#     for col in range(cols):
#         for channel in range(channels):
#             # Feed the whole array and retrieving the pixel value probabilities for the next
#             # pixel.
#             probs = pixel_cnn.predict(pixels)[:, row, col, channel]
#             # Use the probabilities to pick pixel values and append the values to the image
#             # frame.
#             pixels[:, row, col, channel] = tf.math.ceil(
#                 probs - tf.random.uniform(probs.shape)
#             )
#
# def deprocess_image(x):
#     # Stack the single channeled black and white image to RGB values.
#     x = np.stack((x, x, x), 2)
#     # Undo preprocessing
#     x *= 255.0
#     # Convert to uint8 and clip to the valid range [0, 255]
#     x = np.clip(x, 0, 255).astype("uint8")
#     return x
#
# # Iterate over the generated images and plot them with matplotlib.
# for i, pic in enumerate(pixels):
#     keras.preprocessing.image.save_img(
#         "generated_image_{}.png".format(i), deprocess_image(np.squeeze(pic, -1))
#     )
#
# display(Image("generated_image_0.png"))
# display(Image("generated_image_1.png"))
# display(Image("generated_image_2.png"))
# display(Image("generated_image_3.png"))
