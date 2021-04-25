import argparse
import json
import os
import tensorflow as tf

from trainer import model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Required arguments
  parser.add_argument('--data_bucket_name',
                      help='GCS location of the training images',
                      required=True)
  parser.add_argument('--checkpoint_path',
                      help='GCS location to save model weights to',
                      required=True)

  # Optional arguments
  parser.add_argument('--job-dir',
                      help='This model ignores this field, but it is required by gcloud',
                      default='junk')
  parser.add_argument('--resolution',
                      help='Final resolution of the output images',
                      type=int,
                      default=128)
  parser.add_argument('--batch_size',
                      help='Training batch size',
                      type=int,
                      default=64)
  parser.add_argument('--latent_size',
                      help='Dimension of the latent space the generator '
                          'input is sampled from',
                      type=int,
                      default=None)
  parser.add_argument('--fmap_base',
                      help='Base value for number of feature maps.',
                      type=int,
                      default=8192)
  parser.add_argument('--fmap_max',
                      help='Max value for number of feature maps.',
                      type=int,
                      default=512)
  parser.add_argument('--fmap_decay',
                      help='Decay value for number of feature maps.',
                      type=float,
                      default=1.0)
  parser.add_argument('--normalize_latents',
                      help='Toggles normalizing the latent vector for the '
                          'generator',
                      type=bool,
                      default=True)
  parser.add_argument('--use_wscale',
                      help='Toggles weight scaling',
                      type=bool,
                      default=True)
  parser.add_argument('--use_pixel_norm',
                      help='Toggles pixelwise normalization in convolutional '
                          'layers',
                      type=bool,
                      default=True)
  parser.add_argument('--use_leaky_relu',
                      help='Toggles using leaky ReLU activation',
                      type=bool,
                      default=True)
  parser.add_argument('--num_channels',
                      help='Number of output channels',
                      type=int,
                      default=3)
  parser.add_argument('--mbstd_group_size',
                      help='Minibatch standard deviation size',
                      type=int,
                      default=4)
  parser.add_argument('--learning_rate',
                      help='Optimizer learning rate',
                      type=float,
                      default=0.001)
  parser.add_argument('--learning_rate_decay',
                      help='Learning rate decay rate',
                      type=float,
                      default=0.8)
  parser.add_argument('--gradient_weight',
                      help='Gradient penalty loss term weight',
                      type=float,
                      default=10.0)
  parser.add_argument('--D_repeat',
                      help='Train batches for the critic per generator '
                          'training batch',
                      type=int,
                      default=1)
  parser.add_argument('--kimage_4x4',
                      help='Number of training images for 4x4 resolution',
                      type=int,
                      default=1000)
  parser.add_argument('--kimage',
                      help='Number of training images for resolutions < 128',
                      type=int,
                      default=2000)
  parser.add_argument('--kimage_large',
                      help='Number of training images for resolutions >= 128',
                      type=int,
                      default=4000)
  parser.add_argument('--data_filename',
                      help='Name of the .zip file with the data in GCP',
                      type=str,
                      default='celeba.zip')
  parser.add_argument('--debug_mode',
                      help='Toggles debug logging',
                      type=bool,
                      default=True)
  parser.add_argument('--print_every_n_batches',
                      help='Logs progress every N training batches',
                      type=int,
                      default=50)
  parser.add_argument('--save_every_n_batches',
                      help='Saves model every N training batches',
                      type=int,
                      default=1000)

  args = parser.parse_args()
  arguments = args.__dict__
  arguments.pop("job_dir", None)
  arguments.pop("job-dir", None)

  model.train(**arguments)
