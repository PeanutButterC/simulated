import numpy as np
import tensorflow as tf
from paths import SIMULATED_PATH
from simulated_data import NaiveDataGenerator, CPCDataGenerator, NaiveCrowdsourcingDataGenerator, CPCCrowdsourcingDataGenerator, ACTIONS
from models import NaiveModel, CPCModel
from omegaconf import OmegaConf
import argparse
import os
from tqdm import tqdm


class NaiveCrowdsourcingProbe(tf.keras.Model):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.rnn = tf.keras.layers.RNN(self.model.lstm_cell, return_sequences=True)
        self.flatten = tf.keras.layers.Flatten()
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(cfg.model.hidden_width, activation='relu'),
            tf.keras.layers.Dense(len(ACTIONS), activation='sigmoid')
        ])
        self.model.trainable = False
        self.rnn.trainable = False

    def call(self, x):
        x = self.model.dense(x)
        x = self.rnn(x)
        x = self.flatten(x)
        return self.decoder(x)


class CPCCrowdsourcingProbe(tf.keras.Model):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(cfg.model.proj_width, activation='relu'),
            tf.keras.layers.Dense(len(ACTIONS), activation='sigmoid')
        ])
        self.model.trainable = False

    def call(self, x):
        x = self.model.encoder_layer(x)
        x, _, _ = self.model.lstm_rnn(x)
        return self.decoder(x)


class CPCCrowdsourcingEncoderProbe(tf.keras.Model):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(cfg.model.proj_width, activation='relu'),
            tf.keras.layers.Dense(len(ACTIONS), activation='sigmoid')
        ])
        self.model.trainable = False

    def call(self, x):
        x = self.model.encoder_layer(x)
        return self.decoder(x)


def main(cfg, args, encoder):
    if cfg.model.name == 'naive':
        probe_gen = NaiveCrowdsourcingDataGenerator(cfg)
        gen = NaiveDataGenerator(cfg)
        model = NaiveModel(cfg)
        probe = NaiveCrowdsourcingProbe(cfg, model)
        model(next(iter(gen.train))[0])
        model.load_weights(f'{SIMULATED_PATH}/outputs/{args.dir}/weights.h5')
    elif cfg.model.name == 'cpc':
        probe_gen = CPCCrowdsourcingDataGenerator(cfg)
        gen = CPCDataGenerator(cfg)
        model = CPCModel(cfg)
        if encoder:
            probe = CPCCrowdsourcingProbe(cfg, model)
        else:
            probe = CPCCrowdsourcingEncoderProbe(cfg, model)
        model(next(iter(gen.train)))
        model.load_weights(f'{SIMULATED_PATH}/outputs/{args.dir}/weights.h5')
    else:
        raise f'{cfg.model.name} not implemented'
    if encoder:
        probe_path = f'{SIMULATED_PATH}/outputs/{args.dir}/crowdsourcing_probe_encoder_weights.h5'
    else:
        probe_path = f'{SIMULATED_PATH}/outputs/{args.dir}/crowdsourcing_probe_weights.h5'
    if os.path.exists(probe_path):
        probe(next(iter(probe_gen.train))[0])
        probe.load_weights(probe_path)
    else:
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        best_prec = 0
        k = 0
        patience = 25
        for epoch in range(999):
            train_loss_avg = tf.keras.metrics.Mean()
            dev_loss_avg = tf.keras.metrics.Mean()
            accuracy_avg = [tf.keras.metrics.BinaryAccuracy() for action in ACTIONS]
            precision_avg = [tf.keras.metrics.Precision() for action in ACTIONS]
            for x_batch_train, y_batch_train in tqdm(probe_gen.train):
                with tf.GradientTape() as tape:
                    mask = tf.cast(tf.math.not_equal(y_batch_train, -1), tf.float32)
                    y_batch_pred = probe(x_batch_train)
                    y_batch_train = tf.cast(y_batch_train, tf.float32) * mask
                    y_batch_pred *= mask
                    loss = loss_fn(y_batch_train, y_batch_pred)
                grads = tape.gradient(loss, probe.trainable_weights)
                optimizer.apply_gradients(zip(grads, probe.trainable_weights))
                train_loss_avg.update_state(loss)
            for x_batch_dev, y_batch_dev in probe_gen.dev:
                mask = tf.cast(tf.math.not_equal(y_batch_dev, -1), tf.float32)
                y_batch_pred = probe(x_batch_dev)
                y_batch_dev = tf.cast(y_batch_dev, tf.float32) * mask
                y_batch_pred *= mask
                loss = loss_fn(y_batch_dev, y_batch_pred)
                dev_loss_avg.update_state(loss)
                for i, action in enumerate(ACTIONS):
                    accuracy_avg[i].update_state(y_batch_dev[:, i, tf.newaxis], y_batch_pred[:, i, tf.newaxis], sample_weight=mask[:, i])
                    precision_avg[i].update_state(y_batch_dev[:, i, tf.newaxis], y_batch_pred[:, i, tf.newaxis], sample_weight=mask[:, i])
            train_loss = train_loss_avg.result()
            dev_loss = dev_loss_avg.result()
            accuracies = [float(acc.result()) for acc in accuracy_avg]
            precisions = [float(prec.result()) for prec in precision_avg]
            acc_macro = np.mean(accuracies)
            prec_macro = np.mean(precisions)
            print(f'Train loss: {train_loss} Dev loss: {dev_loss} Accuracy (macro): {acc_macro} Precision (macro): {prec_macro}', flush=True)
            if prec_macro > best_prec:
                k = 0
                best_prec = prec_macro
                print(f'Improved, save to {probe_path}', flush=True)
                probe.save_weights(probe_path)
            else:
                k += 1
                if k >= patience:
                    break


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
