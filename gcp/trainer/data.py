import tensorflow as tf


def load_image_dataset(image_dir, image_size, batch_size):
  """Load the images into a dataset from the given path."""
  return tf.keras.preprocessing.image_dataset_from_directory(
      'data',
      label_mode=None,
      batch_size=batch_size,
      interpolation='nearest',
      image_size=(image_size, image_size)).map(augment_and_scale)


def augment_and_scale(x):
  """Image augmentation and scaling."""
  x = tf.image.random_flip_left_right(x)
  return tf.cast(x, tf.float32) / 255.0


def images_as_np_iter(image_dir, image_size, batch_size):
  """Create a NumPy iterator out of the images in the training set."""
  X_train_ds = load_image_dataset(image_dir, image_size, batch_size)
  return X_train_ds.as_numpy_iterator()


def training_data(image_dir, image_size, batch_size):
  """Return the training set iterator, which can be used for multiple training epochs."""
  X_train = images_as_np_iter(image_dir, image_size, batch_size)
  while True:
    try:
      X_batch = next(X_train)
      if len(X_batch) != batch_size:
        X_train = images_as_np_iter(image_dir, image_size, batch_size)
        X_batch = next(X_train)
      yield X_batch
    except StopIteration:
      X_train = images_as_np_iter(image_dir, image_size, batch_size)
