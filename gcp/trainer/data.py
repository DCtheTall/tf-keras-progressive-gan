import tensorflow as tf


def load_image_dataset(image_dir, batch_size, image_size):
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
