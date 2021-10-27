import tensorflow as tf
from paths import SIMULATED_PATH
from simulated_data import SimulatedDataGenerator, ProbeDataGenerator
from omegaconf import OmegaConf
import argparse
import os
import quantization
from tqdm import tqdm
import data_analysis


class QuantizedProbe(tf.keras.Model):
    def __init__(self, hidden_width, pos_voxels, rot_voxels):
        super().__init__()
        self.target_pos_embedding = tf.keras.layers.Embedding(pos_voxels, hidden_width)
        self.target_rot_embedding = tf.keras.layers.Embedding(rot_voxels, hidden_width)
        self.hand_pos_embedding = tf.keras.layers.Embedding(pos_voxels, hidden_width)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        tp, tr, hp = tf.unstack(x, axis=-1)
        tp = self.target_pos_embedding(tp)
        tr = self.target_rot_embedding(tr)
        hp = self.hand_pos_embedding(hp)
        x = tf.concat([tp, tr, hp], axis=-1)
        return self.dense(x)


def main(cfg):
    gen = SimulatedDataGenerator.from_config(cfg)
    probe_gen = ProbeDataGenerator(gen)
    if cfg.quantized:
        meta = quantization.get_meta(cfg.quantile_size)
        model = QuantizedProbe(cfg.hidden_width, meta['pos_voxels'], meta['rot_voxels'])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    probe_path = f'{SIMULATED_PATH}/outputs/{args.dir}/probe_weights_naive.h5'
    if os.path.exists(probe_path):
        model(next(iter(probe_gen.train))[0])
        model.load_weights(probe_path)
    else:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=cfg.patience, verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(probe_path, save_best_only=True, verbose=1)
        callbacks = [early_stopping, checkpoint]
        model.fit_generator(probe_gen.train, validation_data=probe_gen.dev, callbacks=callbacks, epochs=999, verbose=1)

    y_true = []
    y_pred = []
    for x, y in tqdm(probe_gen.dev):
        y_ = model(x)
        y_ = tf.math.argmax(y_, axis=-1)
        y = tf.reshape(y, (-1,))
        y_ = tf.reshape(y_, (-1,))
        y_true.append(y)
        y_pred.append(y_)
    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)

    data_analysis.eval(y_true, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    path = f'{SIMULATED_PATH}/outputs/{args.dir}/.hydra/config.yaml'
    cfg = OmegaConf.load(path)
    main(cfg)
