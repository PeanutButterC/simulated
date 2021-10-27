from paths import SIMULATED_PATH
from simulated_data import NaiveDataGenerator, CPCDataGenerator
from models import NaiveModel, CPCModel
import hydra
from omegaconf import DictConfig
from losses import DiscountedMSE, contrastive_loss
from metrics import next_timestep_mae
import os
import tensorflow as tf
from tqdm import tqdm


def train_naive(gen, cfg):
    model = NaiveModel(cfg)
    loss = DiscountedMSE(cfg)
    metrics = ['mae', next_timestep_mae]
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=cfg.patience, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{os.getcwd()}/weights.h5', verbose=1, save_best_only=True, save_weights_only=True)
    callbacks = [early_stopping, checkpoint]
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    model.fit(gen.train, validation_data=gen.dev, callbacks=callbacks, epochs=999, verbose=1)


def train_cpc(gen, cfg, model):
    optimizer = tf.keras.optimizers.Adam()
    k = 0
    prev_loss = float('inf')
    weights_path = f'{os.getcwd()}/weights.h5'
    for epoch in range(99999):
        train_loss_avg = tf.keras.metrics.Mean()
        dev_loss_avg = tf.keras.metrics.Mean()
        print(f'\nEpoch {epoch}', end='', flush=True)
        for batch in tqdm(gen.train):
            with tf.GradientTape() as tape:
                z_i, z_j = model(batch, training=True)
                loss, _, _ = contrastive_loss(z_i, z_j, cfg.model.temperature)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                train_loss_avg.update_state(loss)
        for batch in tqdm(gen.dev):
            z_i, z_j = model(batch, training=False)
            loss, _, _ = contrastive_loss(z_i, z_j, cfg.model.temperature)
            dev_loss_avg.update_state(loss)
        train_loss = train_loss_avg.result()
        dev_loss = dev_loss_avg.result()
        print(f'\nTrain Loss: {train_loss} Dev Loss: {dev_loss}', flush=True)
        if dev_loss < prev_loss:
            k = 0
            print(f'Loss improved from {prev_loss} to {dev_loss}, saving to {weights_path}', flush=True)
            model.save_weights(weights_path)
        else:
            k += 1
            print(f'Loss hasn\'t improved for {k} epoch(s)', flush=True)
            if k >= cfg.patience:
                print('Quitting', flush=True)
                break
        prev_loss = dev_loss


@hydra.main(config_path=f'{SIMULATED_PATH}/conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print(cfg)
    if cfg.model.name == 'naive':
        gen = NaiveDataGenerator(cfg)
        train_naive(gen, cfg)
    elif cfg.model.name == 'cpc':
        model = CPCModel(cfg)
        gen = CPCDataGenerator(cfg)
        train_cpc(gen, cfg, model)
    else:
        print(f'unknown model {cfg.model.name}')


if __name__ == '__main__':
    main()
