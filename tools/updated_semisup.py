import numpy as np
import tensorflow as tf
def create_input(input_images, input_labels, batch_size):

    if input_labels is not None:
        training_dataset = tf.data.Dataset.from_tensor_slices((input_images, input_labels)).batch(batch_size)
    else:
        training_dataset = tf.data.Dataset.from_tensor_slices(input_images).batch(batch_size)
    #iterator = training_dataset.shuffle(len(input_labels)).make_one_shot_iterator()
    #iterator = training_dataset.shuffle(len(input_images)).make_initializable_iterator()

    #return iterator.get_next()
    #return iterator
    return training_dataset.shuffle(len(input_images))

def create_per_class_inputs(images_by_class, n_per_class, class_labels=None):

    if class_labels is None:
        class_labels = np.arange(len(images_by_class))

    batch_images, batch_labels = [], []

    class_datasets = []

    for images, label in zip (images_by_class, class_labels):
        labels = tf.fill([len(images)], label)
        # images, labels = create_input(images, labels, n_per_class)
        
        training_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(n_per_class).shuffle(len(images))

        #iterator = create_input(images, labels, n_per_class)
        data = create_input(images, labels, n_per_class)

        class_datasets.append(data)

        #batch_images.append(images)
        #batch_labels.append(labels)

    zipped = tf.data.Dataset.zip(tuple(class_datasets))

    #return tf.concat(batch_images, 0), tf.concat(batch_labels, 0)
    return zipped.repeat()
