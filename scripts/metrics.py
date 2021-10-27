import tensorflow as tf


def next_timestep_mae(y_true, y_pred):
    y_true = y_true[:, 0:1, :]
    y_pred = y_pred[:, 0:1, :]
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)


def next_timestep_accuracy(y_true, y_pred):
    y_true = y_true[:, 0:1]
    y_pred = y_pred[:, 0:1, :]
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
