# inspired by:
# - https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html
# - https://flax.readthedocs.io/en/latest/mnist_tutorial.html

import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

tf.random.set_seed(0)  # Set the random seed for reproducibility.

train_steps = 1200
eval_every = 200
batch_size = 32

#####
# prepare datasets
#####
train_ds = tfds.load('mnist', split='train')
test_ds = tfds.load('mnist', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)

len(train_ds) # 60000

test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)

len(test_ds) # 10000

train_ds = train_ds.repeat().shuffle(1024)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

#####
# model
#####

params = {
}