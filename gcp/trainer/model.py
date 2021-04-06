import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def n_filters(stage, fmap_base, fmap_max, fmap_decay):
  """Get the number of filters in a convolutional layer."""
  return int(min(fmap_max, fmap_base / 2.0 ** (stage * fmap_decay)))


def pixelwise_feature_norm(x, epsilon=1e-8):
  """Pixelwise feature normalization for the output of convolutional layers."""
  return x / K.sqrt(
      K.mean(K.square(x), axis=len(x.shape)-1, keepdims=True) + epsilon)


def layer_init_stddev(shape, gain=np.sqrt(2)):
  """Get the He initialization scaling term."""
  fan_in = np.prod(shape[:-1])
  return gain / np.sqrt(fan_in)


def Dense(x, units,
          use_wscale=False,
          gain=np.sqrt(2),
          name=None):
  """Build a densely connected layer."""
  if len(x.shape) > 2:
    x = K.reshape(x, shape=(-1, np.prod(x.shape[1:])))
  stddev = layer_init_stddev([x.shape[1], units], gain=gain)
  if use_wscale:
    weight_init = tf.keras.initializers.RandomNormal()
    x = tf.keras.layers.Dense(units, activation=None,
                              kernel_initializer=weight_init,
                              use_bias=False,
                              name=name)(x)
    x = tf.keras.layers.Lambda(
        lambda x: x * K.constant(stddev, dtype='float32'))(x)
    return x
  weight_init = tf.keras.initializers.RandomNormal(0.0, stddev)
  return tf.keras.layers.Dense(units, activation=None,
                               kernel_initializer=weight_init,
                               use_bias=False,
                               name=name)(x)


class Bias(tf.keras.layers.Layer):
  """Custom bias layer for applying the bias after weight scaling."""
  def __init__(self, shape, *args, **kwargs):
    super(Bias, self).__init__(*args, **kwargs)
    self.shape = shape
    self._config = {'shape': shape}
    self.b = self.add_weight('bias',
                             shape=shape,
                             initializer='zeros',
                             trainable=True)

  def call(self, x):
    """Call method for functional API."""
    return x + self.b


def Conv2D(x, filters, kernel,
           gain=np.sqrt(2),
           use_wscale=False,
           name=None,
           **unused_kwargs):
  """Build a 2D convolutional layer."""
  stddev = layer_init_stddev([kernel, kernel, x.shape[1], filters], gain=gain)
  if use_wscale:
    weight_init = tf.keras.initializers.RandomNormal()
    x = tf.keras.layers.Conv2D(filters, kernel,
                               strides=(1, 1),
                               padding='same',
                               kernel_initializer=weight_init,
                               use_bias=False,
                               activation=None,
                               name=name)(x)
    x = tf.keras.layers.Lambda(
        lambda x: x * K.constant(stddev, dtype='float32'))(x)
    return x
  weight_init = tf.keras.initializers.RandomNormal(0.0, stddev)
  return tf.keras.layers.Conv2D(filters, kernel,
                                strides=(1, 1),
                                padding='same',
                                kernel_initializer=weight_init,
                                use_bias=False,
                                activation=None,
                                name=name)(x)


class Lerp(tf.keras.layers.Layer):
  """A linear interpolation layer for fading in higher resolutions."""

  def __init__(self, t, *args, **kwargs):
    super(Lerp, self).__init__(*args, **kwargs)
    self.t = t

  def call(self, a, b, *args, **kwargs):
    """Call method for functional API."""
    return a + (b - a) * self.t


def resolution_label(res_log2):
  """Create an image resolution label."""
  return '{}x{}'.format(1 << res_log2, 1 << res_log2)


def G_block(x, res_log2, n_filters,
            use_pixel_norm=False,
            use_wscale=False,
            use_leaky_relu=False,
            **unused_kwargs):
  """Build a block of the generator."""
  pn = lambda x: pixelwise_feature_norm(x) if use_pixel_norm else x
  act = (tf.keras.layers.LeakyReLU() if use_leaky_relu
         else tf.keras.layers.Activation('relu'))
  
  nf = n_filters(res_log2 - 1)

  if res_log2 == 2:
    # Start with dense layer.
    units = nf << 4
    x = Dense(x, units, gain=np.sqrt(2)/4, use_wscale=use_wscale,
              name='G_dense_head')
    x = K.reshape(x, [-1, 4, 4, nf])
    x = Bias([1, 1, nf], name='G_dense_head_bias')(x)
    x = pn(act(x))
    # Then the first convolutional layer.
    x = Conv2D(x, filters=nf, kernel=3, use_wscale=use_wscale, name='4x4_conv')
    x = Bias([1, 1, nf], name='4x4_conv_bias')(x)
    return pn(act(x))
  res_label = resolution_label(res_log2)
  # Upsample the input.
  x = tf.keras.layers.UpSampling2D()(x)

  # Now 2 convolutional layers.
  x = Conv2D(x, filters=nf, kernel=3, use_wscale=use_wscale,
              name='{}_conv0'.format(res_label))
  x = Bias([1, 1, nf], name='{}_conv0_bias'.format(res_label))(x)
  x = pn(act(x))

  x = Conv2D(x, filters=nf, kernel=3, use_wscale=use_wscale,
              name='{}_conv1'.format(res_label))
  x = Bias([1, 1, nf], name='{}_conv1_bias'.format(res_label))(x)
  return pn(act(x))


def ToRGB(x, channels,
          use_wscale=False,
          name=None,
          **unused_kwargs):
  """Convert a 4D tensor into the RGB color space."""
  x = Conv2D(x, filters=channels, kernel=1, gain=1, use_wscale=use_wscale,
             name=name)
  bias_name = None if name is None else (name + '_bias')
  x = Bias([1, 1, channels], name=bias_name)(x)
  return tf.keras.layers.Activation('sigmoid')(x)


def G(latent_size=None,  # Dimensionality of latent space.
      fmap_base=8192,
      fmap_max=512,  # Max filters in each conv layer.
      fmap_decay=1.0,
      normalize_latents=True,  # Pixelwise normalize latent vector.
      use_wscale=True,  # Scale the weights with He init at runtime.
      use_pixel_norm=True,  # Use pixelwise normalization.
      use_leaky_relu=True,  # True = use LeakyReLU, False = use ReLU.
      num_channels=3,  # Number of output channels.
      resolution=64,  # Resolution of the output.
      **unused_kwargs):
  """Build the generator networks for each size."""
  if latent_size is None:
    latent_size = min(fmap_base, fmap_max)

  partial_nfilters = lambda n: n_filters(n, fmap_base, fmap_max, fmap_decay)

  opts = {
      'use_wscale': use_wscale,
      'use_pixel_norm': use_pixel_norm,
      'use_leaky_relu': use_leaky_relu,
  }

  # We can set the value of this during training with a callback.
  alpha = K.variable(0.0, dtype='float32', name='alpha')

  models = {}
  resolution_log2 = int(np.log2(resolution))

  for max_res_log2 in range(2, resolution_log2 + 1):
    latents_in = tf.keras.layers.Input(shape=(latent_size,), name='latents_in')
    x = latents_in
    if normalize_latents:
      x = pixelwise_feature_norm(x)
    x = G_block(x, 2, partial_nfilters, **opts)
    img = ToRGB(x, num_channels, **opts, name='4x4_to_rgb')

    if max_res_log2 == 2:
      models['4x4'] = tf.keras.models.Model(latents_in, img)
      continue

    img_prev = img

    for res_log2 in range(3, max_res_log2 + 1):
      x = G_block(x, res_log2, partial_nfilters, **opts)
      if res_log2 == (max_res_log2 - 1):
        img_prev = ToRGB(x, num_channels, **opts,
                         name='{}_to_rgb'.format(resolution_label(res_log2)))
      if res_log2 == max_res_log2:
        img = ToRGB(x, num_channels, **opts,
                    name='{}_to_rgb'.format(resolution_label(res_log2)))
        img = Lerp(alpha)(tf.keras.layers.UpSampling2D()(img_prev), img)

    label = resolution_label(max_res_log2)
    models[label] = tf.keras.models.Model(latents_in, [img, img_prev])

  return models, alpha


def MinibatchStddev(x, group_size):
  """Contatenate the stddev of the minibatch to the output of a conv layer."""
  y = K.reshape(x, [group_size, -1, *x.shape[1:]])
  y = K.cast(y, 'float32')
  y -= K.mean(y, axis=0, keepdims=True)
  y = K.mean(K.square(y), axis=0)
  y = K.cast(y, x.dtype)
  y = K.tile(y, [group_size, 1, 1, 1])
  return K.concatenate([x, y], axis=1)


def Downscale2D(x):
  """Downscale layer scales an image down 2x."""
  return tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid',
                                          data_format='channels_last')(x)


def D_block(x, res_log2, n_filters,
            mbstd_group_size=4,
            use_wscale=False,
            **unused_kwargs):
  """Build a block of the discriminator conv net."""
  act = tf.keras.layers.LeakyReLU()
  nf = n_filters(res_log2 - 1)
  if res_log2 == 2:
    # Apply minibatch stddev layer.
    x = MinibatchStddev(x, mbstd_group_size)

    # Then a convolutional layer.
    x = Conv2D(x, filters=nf, kernel=3, use_wscale=use_wscale, name='4x4_conv')
    x = Bias([1, 1, nf], name='4x4_conv_bias')(x)
    x = act(x)

    # Now we add the dense head to the discriminator.
    nf = n_filters(0)
    x = tf.keras.layers.Flatten(data_format='channels_last')(x)
    x = Dense(x, nf, use_wscale=use_wscale, name='D_dense_head')
    x = Bias([nf], name='D_dense_head_bias')(x)
    x = act(x)
    x = Dense(x, 1, gain=1, use_wscale=use_wscale, name='D_logit')
    x = Bias([1], name='D_logit_bias')(x)
  else:
    label = resolution_label(res_log2)

    # First convolutional layer.
    x = Conv2D(x, filters=nf, kernel=3, use_wscale=use_wscale,
               name='{}_conv0'.format(label))
    x = Bias([1, 1, nf], name='{}_conv0_bias'.format(label))(x)
    x = act(x)

    # Second convolutional layer.
    nf = n_filters(res_log2 - 2)
    x = Conv2D(x, filters=nf, kernel=3, use_wscale=use_wscale,
               name='{}_conv1'.format(label))
    x = Bias([1, 1, nf], name='{}_conv1_bias'.format(label))(x)
    x = act(x)
    
    # And downscale.
    x = Downscale2D(x)
  return x


def FromRGB(x, filters,
            use_wscale=False,
            name=None,
            **unused_kwargs):
  """First covolutional layer after an RGB layer."""
  x = Conv2D(x, filters=filters, kernel=1, use_wscale=use_wscale, name=name)
  bias_name = None if name is None else name + '_bias'
  x = Bias([1, 1, filters], name=bias_name)(x)
  return tf.keras.layers.LeakyReLU()(x)


def D(num_channels=3,  # Number of channels images have.
      resolution=64,  # Max image resolution.
      fmap_base=8192,
      fmap_max=512,  # Max filters in each conv layer.
      fmap_decay=1.0,
      use_wscale=True,  # Scale the weights with He init at runtime.
      mbstd_group_size=4, # Group size for minibatch stddev layer.
      **unused_kwargs):
  """Build the discriminator networks for each size."""
  partial_nfilters = lambda n: n_filters(n, fmap_base, fmap_max, fmap_decay)
  resolution_log2 = int(np.log2(resolution))
  opts = {
      'use_wscale': use_wscale,
      'mbstd_group_size': mbstd_group_size,
  }

  # We can set the value of this during training with a callback.
  alpha = K.variable(0.0, dtype='float32', name='alpha')
  img_ins = {
      resolution_label(res_log2): tf.keras.layers.Input(
          shape=(1 << res_log2, 1 << res_log2, num_channels))
      for res_log2 in range(2, resolution_log2 + 1)
  }
  outputs = {}

  for max_res_log2 in range(2, resolution_log2 + 1):
    label = resolution_label(max_res_log2)
    x = img_ins[label]
    x = FromRGB(x, filters=partial_nfilters(max_res_log2 - 1),
                name='{}_from_rgb'.format(label), **opts)
    for res_log2 in range(max_res_log2, 2, -1):
      x = D_block(x, res_log2, partial_nfilters, **opts)
      if res_log2 == max_res_log2:
        prev_label = resolution_label(res_log2 - 1)
        prev_img = img_ins[prev_label]
        y = FromRGB(prev_img, partial_nfilters(res_log2 - 2),
                    name='{}_from_rgb'.format(prev_label), **opts)
        x = Lerp(alpha)(y, x)
    outputs[label] = D_block(x, 2, partial_nfilters, **opts)

  models = {}
  for res_log2 in range(2, resolution_log2 + 1):
    label = resolution_label(res_log2)
    ins = [img_ins[label]]
    if res_log2 > 2:
      ins.append(img_ins[resolution_label(res_log2 - 1)])
    models[label] = tf.keras.models.Model(ins, outputs[label])
  
  return models, alpha


class WGANGP(tf.keras.models.Model):
  """
  A single WGAN-GP network.

  This implements the training objective for a single resolution
  of ProGAN. This model is applied iteratively over each resolution
  with a fresh optimizer. The motivation of ProGAN is to treat each
  resolution as an entirely different learning task, and uses the
  pretrained lower layers for transfer learning.

  In order to properly implement the gradient penalty loss term
  with TensorFlow 2.x, we need to extend the Keras model with
  our own custom training method. Since we have to use this model
  extension, we can also encapsulate the "fading in" the next
  resolutions.
  
  """

  def __init__(self,
               resolution,
               G,
               D,
               G_optimizer,
               D_optimizer,
               batch_size,
               latent_size,
               G_alpha=None,
               D_alpha=None,
               gradient_weight=10.0,
               D_repeat=1,
               *args, **kwargs):
    super(WGANGP, self).__init__(*args, **kwargs)
    self.resolution = resolution
    self.G = G
    self.D = D
    self.G_optimizer = G_optimizer
    self.D_optimizer = D_optimizer
    self.batch_size = batch_size
    self.latent_size = latent_size
    self.gradient_weight = gradient_weight

    self.G_alpha = G_alpha
    self.D_alpha = D_alpha

    self.D_repeat = D_repeat

  def compute_D_loss(self, real_imgs):
    """Compute discriminator loss terms."""

    # Generate the image at lower resolution for layer fading.
    latents_in = np.random.normal(size=(self.batch_size, self.latent_size))
    if self.resolution > 4:
      fake_imgs = self.G(latents_in)
      interp_imgs = [self.interpolate_imgs(real_imgs[i], fake_imgs[i])
                     for i in range(2)]
    else:
      fake_imgs = [self.G(latents_in)]
      interp_imgs = [self.interpolate_imgs(real_imgs[0], fake_imgs[0])]
    
    real_pred = self.D(real_imgs)
    fake_pred = self.D(fake_imgs)

    real_loss = tf.reduce_mean(real_pred)
    fake_loss = -tf.reduce_mean(fake_pred)
    gp_loss = self.gradient_penalty(interp_imgs)

    return real_loss, fake_loss, gp_loss

  def interpolate_imgs(self, real_img, fake_img):
    """Interpolate real and fake images for GP loss calculation."""
    w = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
    return (w * real_img) + ((1.0 - w) * fake_img)

  def gradient_penalty(self, interp_imgs):
    """Compute gradient penalty loss."""
    with tf.GradientTape() as tape:
      tape.watch(interp_imgs)
      interp_pred = self.D(interp_imgs)
    grads = tape.gradient(interp_pred, interp_imgs)[0]
    ddx = K.sqrt(K.sum(tf.square(grads), axis=np.arange(1, len(grads.shape))))
    loss = tf.reduce_mean(tf.square(1.0 - ddx))
    return self.gradient_weight * loss

  def compute_G_loss(self):
    """Compute G loss."""
    latents_in = np.random.normal(size=(self.batch_size, self.latent_size))

    # For fading in the new layer.
    if self.resolution > 4:
      fake_imgs = self.G(latents_in)
    else:
      fake_imgs = [self.G(latents_in)]

    fake_pred = self.D(fake_imgs)
    return tf.reduce_mean(fake_pred)

  def compute_D_gradients(self, real_imgs, print_loss=False):
    """Compute the discriminator loss gradients."""
    with tf.GradientTape() as tape:
      D_loss_real, D_loss_fake, D_loss_gp = self.compute_D_loss(real_imgs)
      D_loss = D_loss_real + D_loss_fake + D_loss_gp

    if print_loss:
      print('D Loss: R: {:04f} F: {:04f} GP: {:04f}'.format(
          D_loss_real, D_loss_fake, D_loss_gp))
    
    return tape.gradient(D_loss, self.D.trainable_variables)

  def compute_G_gradients(self, print_loss=False):
    """Compute the generator loss gradients."""
    with tf.GradientTape() as tape:
      G_loss = self.compute_G_loss()

    if print_loss:
      print('G Loss: {:04f}'.format(G_loss))

    return tape.gradient(G_loss, self.G.trainable_variables)

  def train_on_batch(self, X_batch, alpha=None, print_loss=False):
    """Train on a single batch of data."""
    if alpha is not None:
      K.set_value(self.G_alpha, alpha)
      K.set_value(self.D_alpha, alpha)

    for _ in range(self.D_repeat):
      D_grads = self.compute_D_gradients(X_batch, print_loss)
      self.D_optimizer.apply_gradients(zip(D_grads, self.D.trainable_variables))
    
    G_grads = self.compute_G_gradients(print_loss)
    self.G_optimizer.apply_gradients(zip(G_grads, self.G.trainable_variables))
