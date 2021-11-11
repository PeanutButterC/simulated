import tensorflow as tf
from paths import SIMULATED_PATH
from models import NaiveModel
from simulated_data import NaiveDataGenerator, NaiveCrowdsourcingDataGenerator, ACTIONS
from omegaconf import OmegaConf


def masked_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true = tf.cast(y_true, tf.float32) * mask
    y_pred = y_pred * mask
    return tf.keras.metrics.binary_crossentropy(y_true, y_pred)


def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true = tf.cast(y_true, tf.float32) * mask
    y_pred = y_pred * mask
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)


class PretrainedProbe(tf.keras.Model):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.rnn = tf.keras.layers.RNN(model.lstm_cell, return_sequences=True)
        self.prediction_head = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(ACTIONS), activation='sigmoid')
        ])
        self.model.trainable = False
        self.rnn.trainable = False

    def call(self, x):
        x = self.model.dense(x)
        x = self.rnn(x)
        return self.prediction_head(x)


def train_pretrained(cfg, fine_tuning=False):
    model = NaiveModel(cfg)
    model_gen = NaiveDataGenerator(cfg)
    model(next(iter(model_gen.train))[0])
    model.load_weights(f'{SIMULATED_PATH}/outputs/naive_batch-size-1024/weights.h5')
    gen = NaiveCrowdsourcingDataGenerator(cfg)
    probe = PretrainedProbe(cfg, model)
    if fine_tuning:
        probe.model.trainable = True
        probe.rnn.trainable = True
        weights_path = 'outputs/pretrained_finetuning.h5'
        lr = 1e-5
    else:
        weights_path = 'outputs/pretrained_probe.h5'
        lr = 1e-4
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=25, verbose=1, monitor='val_loss')
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    probe.compile(optimizer=optimizer, loss=masked_crossentropy, metrics=[masked_accuracy, tf.keras.metrics.Precision()])
    probe.fit(gen.train, validation_data=gen.dev, epochs=999, callbacks=callbacks, verbose=1)


def train_supervised(cfg):
    gen = NaiveCrowdsourcingDataGenerator(cfg)
    probe = tf.keras.models.Sequential([
        tf.keras.layers.Dense(cfg.model.hidden_width, activation='relu'),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(len(ACTIONS), activation='sigmoid')
    ])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('outputs/supervised.h5', monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=25, verbose=1, monitor='val_loss')
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    probe.compile(optimizer=optimizer, loss=masked_crossentropy, metrics=[masked_accuracy, tf.keras.metrics.Precision()])
    probe.fit(gen.train, validation_data=gen.dev, epochs=999, callbacks=callbacks, verbose=1)


def train_direct_model(cfg):
    gen = NaiveCrowdsourcingDataGenerator(cfg)
    probe = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(ACTIONS), activation='sigmoid')
    ])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('outputs/direct.h5', monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=25, verbose=1, monitor='val_loss')
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    probe.compile(optimizer=optimizer, loss=masked_crossentropy, metrics=[masked_accuracy, tf.keras.metrics.Precision()])
    probe.fit(gen.train, validation_data=gen.dev, epochs=999, callbacks=callbacks, verbose=1)


if __name__ == '__main__':
    cfg = OmegaConf.load(f'{SIMULATED_PATH}/outputs/naive_batch-size-1024/.hydra/config.yaml')
    cfg.model.batch_size = 16
    # train_direct_model(cfg)
    # train_supervised(cfg)
    train_pretrained(cfg, fine_tuning=True)
