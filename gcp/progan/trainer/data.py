import io
import os
import tensorflow as tf
import zipfile

from google.cloud import storage
from tensorflow.python.platform import gfile


DATA_DIR = 'celeba_data/'


def load_image_dataset(bucket_name, data_filename, image_size, batch_size):
  """Load the images into a dataset from the given path."""
  if not os.path.exists(DATA_DIR):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(data_filename)
    zipbytes = io.BytesIO(blob.download_as_string())

    if not zipfile.is_zipfile(zipbytes):
      raise ValueError('Expected .zip file')

    with zipfile.ZipFile(zipbytes, 'r') as zf:
      os.mkdir(DATA_DIR)
      zf.extractall(DATA_DIR)

  ds = tf.keras.preprocessing.image_dataset_from_directory(
      DATA_DIR,
      label_mode=None,
      batch_size=batch_size,
      interpolation='nearest',
      image_size=(image_size, image_size)).map(lambda x: augment(x, image_size))
  return ds.map(augment_and_scale).repeat()


def augment_and_scale(x):
  """Image augmentation and scaling."""
  x = tf.image.random_flip_left_right(x)
  return tf.cast(x, tf.float32) / 255.0


def images_as_np_iter(bucket_name, data_filename, image_size, batch_size):
  """Create a NumPy iterator out of the images in the training set."""
  X_train_ds = load_image_dataset(bucket_name,
                                  data_filename,
                                  image_size,
                                  batch_size)
  return X_train_ds.as_numpy_iterator()


def training_data(bucket_name, data_filename, image_size, batch_size):
  """Return the training set iterator, which can be used for multiple training epochs."""
  X_train = images_as_np_iter(bucket_name,
                              data_filename,
                              image_size,
                              batch_size)
  while True:
     X_batch = next(X_train)
     if len(X_batch) != batch_size:
       X_batch = next(X_train)
     yield X_batch
