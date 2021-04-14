import tensorflow as tf
from tensorflow.python.platform import gfile


# Cache images in memory so we don't have to load them from GCS each time.
image_cache = {}
# Cache filenames in the directory too since getting all of them is also a
# bit time consuming.
filelist = []


def load_image_dataset(image_dir, image_size, batch_size):
  """Load the images into a dataset from the given path."""
  def data_gen():
    return data_generator(
        image_dir, image_size, image_cache, filelist)
  ds = tf.data.Dataset.from_generator(
      data_gen, output_signature=tf.TensorSpec(
          shape=(image_size, image_size, 3), dtype=tf.float32))
  ds = ds.shuffle(batch_size, reshuffle_each_iteration=True)
  return ds.batch(batch_size).map(augment_and_scale).repeat()


def data_generator(image_dir, image_size, image_cache, filelist):
  """Generator for loading image data."""
  if not filelist:
    filelist.extend([fname for gfile.ListDirectory(image_dir)])
  for fname in filelist:
    if fname in image_cache:
        yield image_cache[fname]
        continue
    img = tf.image.decode_png(
        tf.io.read_file(os.path.join(image_dir, fname))).numpy()
    img = tf.image.resize(img, (image_size,) * 2, method='nearest')
    image_cache[fname] = img
    yield img


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
     X_batch = next(X_train)
     if len(X_batch) != batch_size:
       X_batch = next(X_train)
     yield X_batch
