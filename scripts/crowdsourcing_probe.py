import tensorflow as tf
from paths import SIMULATED_PATH
from simulated_data import NaiveDataGenerator, CPCDataGenerator, NaiveCrowdsourcingDataGenerator, CPCCrowdsourcingDataGenerator, ACTIONS
from models import NaiveModel, CPCModel
from omegaconf import OmegaConf
import argparse
import os


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
        return self.decoder(self.flatten(x))


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


def main(cfg, linear):
    if cfg.model.name == 'naive':
        probe_gen = NaiveCrowdsourcingDataGenerator(cfg)
        if args.linear:
            probe = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(cfg.model.hidden_width, activation='relu'),
                tf.keras.layers.Dense(len(ACTIONS), activation='sigmoid')
            ])
        else:
            gen = NaiveDataGenerator(cfg)
            model = NaiveModel(cfg)
            probe = NaiveCrowdsourcingProbe(cfg, model)
            model(next(iter(gen.train))[0])
            model.load_weights(f'{SIMULATED_PATH}/outputs/{args.dir}/weights.h5')
    elif cfg.model.name == 'cpc':
        probe_gen = CPCCrowdsourcingDataGenerator(cfg)
        if args.linear:
            probe = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(cfg.model.proj_width, activation='relu'),
                tf.keras.layers.Dense(len(ACTIONS), activation='sigmoid')
            ])
        else:
            gen = CPCDataGenerator(cfg)
            model = CPCModel(cfg)
            probe = CPCCrowdsourcingProbe(cfg, model)
            model(next(iter(gen.train)))
            model.load_weights(f'{SIMULATED_PATH}/outputs/{args.dir}/weights.h5')
    else:
        raise f'{cfg.model.name} not implemented'
    if linear:
        probe_path = f'{SIMULATED_PATH}/outputs/{args.dir}/crowdsourcing_linear_probe_weights.h5'
    else:
        probe_path = f'{SIMULATED_PATH}/outputs/{args.dir}/crowdsourcing_probe_weights.h5'
    if os.path.exists(probe_path):
        probe(next(iter(probe_gen.train))[0])
        probe.load_weights(probe_path)
    else:
        probe.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-3, patience=cfg.patience, verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(probe_path, save_best_only=True, verbose=1)
        callbacks = [early_stopping, checkpoint]
        probe.fit(probe_gen.train, validation_data=probe_gen.dev, callbacks=callbacks, epochs=999, verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--linear', action='store_true')
    args = parser.parse_args()
    path = f'{SIMULATED_PATH}/outputs/{args.dir}/.hydra/config.yaml'
    cfg = OmegaConf.load(path)
    cfg.model.batch_size = 10
    print(cfg)
    main(cfg, args.linear)
