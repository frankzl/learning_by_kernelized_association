import numpy as np
import tensorflow as tf


def create_input(input_images, input_labels, batch_size):
    if input_labels is not None:
        training_dataset = tf.data.Dataset.from_tensor_slices((input_images, input_labels)).batch(batch_size)
    else:
        training_dataset = tf.data.Dataset.from_tensor_slices(input_images).batch(batch_size)
    return training_dataset.shuffle(len(input_images))


def create_per_class_inputs(images_by_class, n_per_class, class_labels=None):
    if class_labels is None:
        class_labels = np.arange(len(images_by_class))

    class_datasets = []

    for images, label in zip (images_by_class, class_labels):
        labels = tf.fill([len(images)], label)

        # training_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(n_per_class, drop_remainder=True).shuffle(n_per_class)
        training_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(n_per_class, drop_remainder=True).shuffle(n_per_class)

        class_datasets.append(training_dataset)

    zipped = tf.data.Dataset.zip(tuple(class_datasets))

    return zipped.repeat()

def create_per_class_inputs_v2(images_by_class, n_per_class, class_labels=None):
    if class_labels is None:
        class_labels = np.arange(len(images_by_class))

    image_datasets = []
    label_datasets = []

    for images, label in zip (images_by_class, class_labels):
        labels = tf.fill([len(images)], label)

        # training_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(n_per_class, drop_remainder=True).shuffle(n_per_class)
        training_dataset = tf.data.Dataset.from_tensor_slices(images).batch(n_per_class, drop_remainder=True).shuffle(n_per_class)
        label_dataset    = tf.data.Dataset.from_tensor_slices(labels).batch(n_per_class, drop_remainder=True)

        image_datasets.append(training_dataset)
        label_datasets.append(label_dataset)

    return tf.data.Dataset.zip(tuple(image_datasets)), tf.data.Dataset.zip(tuple(label_datasets)).repeat()
