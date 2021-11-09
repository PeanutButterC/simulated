import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from paths import SIMULATED_PATH
from simulated_data import NaiveDataGenerator, NaiveProbeDataGenerator, CPCDataGenerator, CPCProbeDataGenerator
from models import NaiveModel, CPCModel
from omegaconf import OmegaConf
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import data_analysis


class NaiveProbe(tf.keras.Model):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.rnn = tf.keras.layers.RNN(self.model.lstm_cell, return_sequences=True)
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(cfg.model.hidden_width, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.trainable = False
        self.rnn.trainable = False

    def call(self, x):
        x = self.model.dense(x)
        x = self.rnn(x)
        return self.decoder(x)


class CPCProbe(tf.keras.Model):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.rnn = tf.keras.layers.RNN(self.model.lstm_cell, return_sequences=True)
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(cfg.model.proj_width, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.trainable = False
        self.rnn.trainable = False

    def call(self, x):
        x = self.model.encoder_layer(x)
        x = self.rnn(x)
        return self.decoder(x)


class CPCEncoderProbe(tf.keras.Model):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(cfg.model.proj_width, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.trainable = False

    def call(self, x):
        x = self.model.encoder_layer(x)
        return self.decoder(x)


def main(cfg, args, encoder=False):
    if cfg.model.name == 'naive':
        probe_gen = NaiveProbeDataGenerator(cfg)
        gen = NaiveDataGenerator(cfg)
        model = NaiveModel(cfg)
        probe = NaiveProbe(cfg, model)
        model(next(iter(gen.train))[0])
    elif cfg.model.name == 'cpc':
        probe_gen = CPCProbeDataGenerator(cfg)
        gen = CPCDataGenerator(cfg)
        model = CPCModel(cfg)
        if encoder:
            probe = CPCEncoderProbe(cfg, model)
        else:
            probe = CPCProbe(cfg, model)
        model(next(iter(gen.train)))
    else:
        raise Exception(f'Unknown model {cfg.model.name}')
    model.load_weights(f'{SIMULATED_PATH}/outputs/{args.dir}/weights.h5')
    if encoder:
        probe_path = f'{SIMULATED_PATH}/outputs/{args.dir}/probe_encoder_weights.h5'
    else:
        probe_path = f'{SIMULATED_PATH}/outputs/{args.dir}/probe_weights.h5'
    if os.path.exists(probe_path):
        probe(next(iter(probe_gen.train))[0])
        probe.load_weights(probe_path)
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()
        best_prec = 0
        k = 0
        patience = 5
        for epoch in range(999):
            train_loss_avg = tf.keras.metrics.Mean()
            dev_loss_avg = tf.keras.metrics.Mean()
            precision_avg = tf.keras.metrics.Precision()
            f1_micro_avg = tfa.metrics.F1Score(num_classes=10, average='micro')
            f1_macro_avg = tfa.metrics.F1Score(num_classes=10, average='macro')
            for x_batch_train, y_batch_train in tqdm(probe_gen.train):
                with tf.GradientTape() as tape:
                    y_batch_pred = probe(x_batch_train)
                    loss = loss_fn(y_batch_train, y_batch_pred)
                grads = tape.gradient(loss, probe.trainable_weights)
                optimizer.apply_gradients(zip(grads, probe.trainable_weights))
                train_loss_avg.update_state(loss)
            for x_batch_dev, y_batch_dev in probe_gen.dev:
                y_batch_pred = probe(x_batch_dev)
                loss = loss_fn(y_batch_dev, y_batch_pred)
                dev_loss_avg.update_state(loss)
                precision_avg.update_state(y_batch_dev, y_batch_pred)
                f1_micro_avg.update_state(y_batch_dev, y_batch_pred)
                f1_macro_avg.update_state(tf.reshape(y_batch_dev, (-1, 10)), tf.reshape(y_batch_pred, (-1, 10)))
            train_loss = train_loss_avg.result()
            dev_loss = dev_loss_avg.result()
            precision = precision_avg.result()
            f1_micro = f1_micro_avg.result()
            f1_macro = f1_macro_avg.result()
            print(f'Train loss: {train_loss} Dev loss: {dev_loss} Precision: {precision} f1 micro: {f1_micro} f1 macro: {f1_macro}', flush=True)
            if precision > best_prec:
                k = 0
                best_prec = precision
                print('Improved')
                probe.save_weights(probe_path)
            else:
                k += 1
                if k >= patience:
                    break
        early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-3, patience=cfg.patience, verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(probe_path, save_best_only=True, verbose=1)
        callbacks = [early_stopping, checkpoint]
        probe.fit(probe_gen.train, validation_data=probe_gen.dev, callbacks=callbacks, epochs=999, verbose=1)

    y_true = []
    y_pred = []
    for x, y in tqdm(probe_gen.dev):
        y_ = probe(x)
        y_ = tf.math.argmax(y_, axis=-1)
        y = tf.reshape(y, (-1,))
        y_ = tf.reshape(y_, (-1,))
        y_true.append(y)
        y_pred.append(y_)
    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)

    data_analysis.eval(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    cm = tf.math.confusion_matrix(y_true, y_pred)
    cm = cm.numpy()
    print(cm)
    cm = cm / np.linalg.norm(cm, axis=0)
    cm_labels = ['Idle', 'Reach', 'Pick', 'Put', 'Retract', 'Throw', 'PrepareHit', 'Hit', 'PreparePush', 'Push']
    ax = sns.heatmap(cm, xticklabels=cm_labels, yticklabels=cm_labels)
    ax.set(xlabel='predicted label', ylabel='true label')
    plt.xticks(rotation=45)
    plt.title(args.dir)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--encoder', action='store_true')
    args = parser.parse_args()
    path = f'{SIMULATED_PATH}/outputs/{args.dir}/.hydra/config.yaml'
    cfg = OmegaConf.load(path)
    cfg.model.batch_size = 64
    print(cfg)
    main(cfg, args, args.encoder)
