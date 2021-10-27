import numpy as np
import tensorflow as tf


class DiscountedMSE(tf.keras.losses.Loss):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def call(self, y_true, y_pred):
        discount = np.power(np.ones(y_true.shape[1]) * self.cfg.model.discount_factor, np.arange(y_true.shape[1])).reshape((1, -1))
        if self.cfg.model.name == 'raw':
            discount = np.tile(discount[:, :, np.newaxis], (1, 1, self.cfg.model.chunk_size))
        discount = tf.convert_to_tensor(discount, dtype=tf.float32)
        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return loss * discount


def contrastive_loss(z_i, z_j, temperature=1.0):
    batch_size = tf.shape(z_i)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    mask = tf.one_hot(tf.range(batch_size), batch_size)
    logits_aa = tf.matmul(z_i, z_i, transpose_b=True) / temperature
    logits_aa -= mask * 1e9
    logits_bb = tf.matmul(z_j, z_j, transpose_b=True) / temperature
    logits_bb -= mask * 1e9
    logits_ab = tf.matmul(z_i, z_j, transpose_b=True) / temperature
    logits_ba = tf.matmul(z_j, z_i, transpose_b=True) / temperature
    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)
    return loss, logits_ab, labels


if __name__ == '__main__':
    batch_size = 6
    hidden_dim = 2
    z_i = tf.constant(
        [[0, 0],
         [1, 1],
         [2, 2],
         [3, 3],
         [4, 4],
         [5, 5]], dtype=tf.float32)
    z_j = tf.constant(
        [[0, 0],
         [9, 9],
         [9, 9],
         [9, 9],
         [9, 9],
         [9, 9]], dtype=tf.float32)
    # z_i = tf.random.normal((batch_size, hidden_dim))
    # z_j = tf.random.normal((batch_size, hidden_dim))
    print(z_i.numpy().astype(int))
    print(z_j.numpy().astype(int))
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    mask = tf.one_hot(tf.range(batch_size), batch_size)
    logits_aa = tf.matmul(z_i, z_i, transpose_b=True)
    logits_aa -= mask * 1e9
    logits_bb = tf.matmul(z_j, z_j, transpose_b=True)
    logits_bb -= mask * 1e9
    logits_ab = tf.matmul(z_i, z_j, transpose_b=True)
    logits_ba = tf.matmul(z_j, z_i, transpose_b=True)
    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ba, logits_bb], 1))
    print(loss_a.numpy())
    print(loss_b.numpy())
    print((loss_a + loss_b).numpy())
    loss = tf.reduce_mean(loss_a + loss_b)
    print(loss)
    # print(cpcloss(z_i, z_j))
