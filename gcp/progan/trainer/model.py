import datetime
import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K

from trainer import data


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
    return a + (b - a) * K.clip(self.t, 0.0, 1.0)


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
  lod_in = K.variable(0.0, dtype='float32', name='lod_in')
  resolution_log2 = int(np.log2(resolution))

  latents_in = tf.keras.layers.Input(shape=(latent_size,), name='latents_in')
  x = latents_in
  if normalize_latents:
    x = pixelwise_feature_norm(x)

  x = G_block(latents_in, 2, partial_nfilters, **opts)
  img_out = ToRGB(x, num_channels, **opts, name='4x4_to_rgb')
  
  for res_log2 in range(3, resolution_log2 + 1):
    lod = resolution_log2 - res_log2
    x = G_block(x, res_log2, partial_nfilters, **opts)
    img = ToRGB(x, num_channels, **opts,
                name='{}_to_rgb'.format(resolution_label(res_log2)))
    img_out = Lerp(lod_in - lod)(img, tf.keras.layers.UpSampling2D()(img_out))
  
  model = tf.keras.models.Model(latents_in, img_out)

  return model, lod_in


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
      resolution=128,  # Max image resolution.
      fmap_base=8192,
      fmap_max=512,  # Max filters in each conv layer.
      fmap_decay=1.0,
      use_wscale=True,  # Scale the weights with He init at runtime.
      mbstd_group_size=4, # Group size for minibatch stddev layer.
      **unused_kwargs):
  """Build the discriminator networks for each size."""
  partial_nfilters = lambda n: n_filters(n, fmap_base, fmap_max, fmap_decay)
  opts = {
    'use_wscale': use_wscale,
    'mbstd_group_size': mbstd_group_size,
  }
  # We can set the value of this during training with a callback.
  lod_in = K.variable(0.0, dtype='float32', name='lod_in')
  resolution_log2 = int(np.log2(resolution))

  img_in = tf.keras.layers.Input(
      shape=(resolution, resolution, num_channels))

  img = img_in
  x = FromRGB(img, filters=partial_nfilters(resolution_log2 - 1),
              name='{}_from_rgb0'.format(resolution_label(resolution_log2)),
              **opts)

  for res_log2 in range(resolution_log2, 2, -1):
    lod = resolution_log2 - res_log2
    x = D_block(x, res_log2, partial_nfilters, **opts)
    img = Downscale2D(img)
    y = FromRGB(img, filters=partial_nfilters(res_log2 - 2),
                name='{}_from_rgb'.format(resolution_label(res_log2)),
                **opts)
    x = Lerp(lod_in - lod)(x, y)

  output = D_block(x, 2, partial_nfilters, **opts)
  model = tf.keras.models.Model(img_in, output)

  return model, lod_in


class WGANGP:
  """
  This class contains a method for training the adversarial model by batch.

  It implements the training objective for the GAN for a particular resolution.
  When one resolution completes training, the optimizers are overwritten with new
  ones with the same parameters, and training begins at the next resolution.

  In order to properly implement gradient penalty loss in TensorFlow 2.x, we
  need to use a custom Keras model for this training objective.

  """

  def __init__(self,
               resolution,
               G,
               D,
               G_lod_in,
               D_lod_in,
               batch_size,
               latent_size,
               learning_rate=0.001,
               learning_rate_decay=0.8,
               gradient_weight=10.0,
               D_repeat=1,
               *args, **kwargs):
    super(WGANGP, self).__init__(*args, **kwargs)
    self.resolution = resolution
    self.G = G
    self.D = D
    self.G_lod_in = G_lod_in
    self.D_lod_in = D_lod_in
    self.batch_size = batch_size
    self.latent_size = latent_size
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.gradient_weight = gradient_weight
    self.D_repeat = D_repeat

    self.cur_resolution = None
    self.G_optimizer = None
    self.D_optimizer = None

    self.G_lod_in = G_lod_in
    self.D_lod_in = D_lod_in

  def init_optimizers(self, resolution):
    """Initialize the optimizers for the models"""
    self.cur_resolution = resolution
    res_log2 = int(np.log2(resolution))
    lr = self.learning_rate * (self.learning_rate_decay ** (res_log2 - 2))
    self.G_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.0, beta_2=0.99,
                                                epsilon=1e-8)
    self.D_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.0, beta_2=0.99,
                                                epsilon=1e-8)

  def compute_D_loss(self, real_imgs):
    """Compute discriminator loss terms."""

    # Generate the image at lower resolution for layer fading.
    latents_in = np.random.normal(size=(self.batch_size, self.latent_size))
    fake_imgs = self.G(latents_in)
    interp_imgs = self.interpolate_imgs(real_imgs, fake_imgs)
    
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
    fake_imgs = self.G(latents_in)
    fake_pred = self.D(fake_imgs)
    return tf.reduce_mean(fake_pred)

  def compute_D_gradients(self, real_imgs, print_loss=False):
    """Compute the discriminator loss gradients."""
    with tf.GradientTape() as tape:
      D_loss_real, D_loss_fake, D_loss_gp = self.compute_D_loss(real_imgs)
      D_loss = D_loss_real + D_loss_fake + D_loss_gp

    if print_loss:
      logging.info('D Loss: R: {:04f} F: {:04f} GP: {:04f}'.format(
          D_loss_real, D_loss_fake, D_loss_gp))
    
    return tape.gradient(D_loss, self.D.trainable_variables)

  def compute_G_gradients(self, print_loss=False):
    """Compute the generator loss gradients."""
    with tf.GradientTape() as tape:
      G_loss = self.compute_G_loss()

    if print_loss:
      logging.info('G Loss: {:04f}'.format(G_loss))

    return tape.gradient(G_loss, self.G.trainable_variables)

  def train_on_batch(self, X_batch, lod_in=None, print_loss=False):
    """Train on a single batch of data."""
    K.set_value(self.G_lod_in, lod_in)
    K.set_value(self.D_lod_in, lod_in)

    for _ in range(self.D_repeat):
      D_grads = self.compute_D_gradients(X_batch, print_loss)
      self.D_optimizer.apply_gradients(zip(D_grads, self.D.trainable_variables))
    
    G_grads = self.compute_G_gradients(print_loss)
    self.G_optimizer.apply_gradients(zip(G_grads, self.G.trainable_variables))

  def save_model(self, export_path):
  """Save the model parameters to GCS."""
    self.G.save(os.path.join(export_path, 'gen/'))
    self.D.save(os.path.join(export_path, 'disc/'))


def compute_lod_in(lod, cur_img, transition_kimg):
  """Compute value for lod_in, the variable that controls fading in new layers."""
  return lod + min(
      1.0, max(0.0, 1.0 - (float(cur_img) / (transition_kimg * 1000))))


def train(resolution=128,
          batch_size=64,
          latent_size=None,
          fmap_base=8192,
          fmap_max=512,  # Max filters in each conv layer.
          fmap_decay=1.0,
          normalize_latents=True,  # Pixelwise normalize latent vector.
          use_wscale=True,  # Scale the weights with He init at runtime.
          use_pixel_norm=True,  # Use pixelwise normalization.
          use_leaky_relu=True,  # True = use LeakyReLU, False = use ReLU.
          num_channels=3,
          mbstd_group_size=4,
          learning_rate=0.001,
          learning_rate_decay=0.8,
          gradient_weight=10.0,
          D_repeat=1,
          kimage_4x4=1000,
          kimage=2000,
          kimage_large=4000,
          data_bucket_name=None,
          data_filename=None,
          checkpoint_path=None,
          debug_mode=False,
          print_every_n_batches=25,
          save_every_n_batches=1000):
  """Training loop for training the GAN up to the provided resolution."""
  resolution_log2 = int(np.log2(resolution))
  if latent_size is None:
    latent_size = min(fmap_base, fmap_max)

  G_model, G_lod_in = G(resolution=resolution,
                        fmap_base=fmap_base,
                        fmap_decay=fmap_decay,
                        fmap_max=fmap_max,
                        normalize_latents=normalize_latents,
                        use_wscale=use_wscale,
                        use_pixel_norm=use_pixel_norm,
                        use_leaky_relu=use_leaky_relu,
                        num_channels=num_channels)
  D_model, D_lod_in = D(resolution=resolution,
                        num_channels=num_channels,
                        fmap_base=fmap_base,
                        fmap_decay=fmap_decay,
                        fmap_max=fmap_max,
                        use_wscale=use_wscale,
                        mbstd_group_size=mbstd_group_size)

  gan = WGANGP(resolution,
               G_model,
               D_model,
               G_lod_in,
               D_lod_in,
               batch_size=batch_size,
               latent_size=latent_size,
               learning_rate=learning_rate,
               learning_rate_decay=learning_rate_decay,
               gradient_weight=gradient_weight,
               D_repeat=D_repeat)

  def debug_log(*args):
    if debug_mode:
      logging.info(*args)
  debug_log('Debug mode enabled!')

  debug_log('Loading dataset...')
  X_train = data.training_data(data_bucket_name,
                               data_filename,
                               resolution,
                               batch_size)
  debug_log('Data loaded successfully!')

  export_path = os.path.join(
      checkpoint_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

  for res_log2 in range(2, resolution_log2 + 1):
    cur_resolution = 1 << res_log2
    debug_log('Training resolution: {}'.format(cur_resolution))
    
    gan.init_optimizers(cur_resolution)

    if res_log2 == 2:
      total_kimg = kimage_4x4
    elif res_log2 >= 7:
      total_kimg = kimage_large
    else:
      total_kimg = kimage
    transition_kimg = total_kimg // 4

    img_count = 0
    n_batches = (total_kimg * 1000) // batch_size

    lod = resolution_log2 - res_log2

    for i in range(1, n_batches + 1):
      img_count += batch_size
      if res_log2 > 2:
        lod_in_batch = compute_lod_in(lod, img_count, transition_kimg)
      else:
        lod_in_batch = lod
      
      X_batch = next(X_train)
      if (i % print_every_n_batches) == 0:
        debug_log('Batch: {} / {}'.format(i, n_batches))
        debug_log('LoD in: {}'.format(lod_in_batch))

      print_loss = debug_mode and (i % print_every_n_batches) == 0
      gan.train_on_batch(X_batch, lod_in=lod_in_batch, print_loss=print_loss)
      
      if (i % save_every_n_batches) == 0 or i == n_batches:
        debug_log('Saving weights...')
        gan.save_model(export_path)
        debug_log('Done.')
