import tensorflow as tf


class NaiveModel(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dense = tf.keras.layers.Dense(self.cfg.model.hidden_width)
        self.lstm_cell = tf.keras.layers.LSTMCell(self.cfg.model.hidden_width)
        self.lstm = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.output_layer = tf.keras.layers.Dense(10)

    def warmup(self, x):
        x = self.dense(x)
        x, *state = self.lstm(x)
        return x, state

    def call(self, x, training=None):
        predictions = []
        x, state = self.warmup(x)
        first_prediction = self.output_layer(x)
        predictions.append(first_prediction)
        for _ in range(1, self.cfg.model.output_width):
            x, state = self.lstm_cell(x, states=state, training=training)
            next_prediction = self.output_layer(x)
            predictions.append(next_prediction)
        predictions = tf.transpose(tf.stack(predictions), [1, 0, 2])
        return predictions


class CPCModel(tf.keras.models.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(cfg.model.filters, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(cfg.model.proj_width, activation='relu')
        ])
        self.encoder_layer = tf.keras.layers.TimeDistributed(self.encoder)
        self.lstm_cell = tf.keras.layers.LSTMCell(cfg.model.proj_width)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)

    def call(self, inputs, training=None):
        o_i = inputs['o_i']
        o_j = inputs['o_j']
        o_i_encoded = self.encoder_layer(o_i)
        z_i, _, _ = self.lstm_rnn(o_i_encoded, training=training)
        z_j = self.encoder(o_j)
        return z_i, z_j
