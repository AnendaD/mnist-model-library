import tensorflow_datasets as tfds
import numpy as np

train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised = True)

def ds_to_np(ds):
    images = []
    labels = []

    for image, label in ds:
        images.append(image.numpy())
        labels. append(label.numpy())
    return  np.array(images), np.array(labels)

train_images, train_labels = ds_to_np(train_ds)
test_images, test_labels = ds_to_np(test_ds)

train_images = train_images.astype(np.float32) /255.0
test_images = test_images.astype(np.float32) /255.0

train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

train_images.tofile("train_images.mat")
test_images.tofile("test_images.mat")
train_labels.tofile("train_labels.mat")
test_labels.tofile("test_labels.mat")

