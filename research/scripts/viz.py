import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from paths import SIMULATED_PATH
from simulated_data import RawDataGenerator, NaiveVisualizationDataGenerator
from models import LSTMRaw
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import seaborn as sns


label_names = [
        'Idle',
        'Reach',
        'Pick',
        'Put',
        'Retract',
        'Throw',
        'PrepareHit',
        'Hit',
        'PreparePush',
        'Push'
    ]


def plot_lines(cfg):
    datagen = RawDataGenerator(cfg)
    model = LSTMRaw(cfg)
    model(next(iter(datagen.train))[0])
    model.load_weights(f'{SIMULATED_PATH}/outputs/{args.dir}/weights.h5')

    gen = NaiveVisualizationDataGenerator(cfg)
    fig, ax = plt.subplots(2, 2)
    i = 0
    for x, y, states in gen.dev:
        predicted = model(x)
        x = x[0, :, 1]
        predicted = predicted[0][0, :, 1]
        y = y[0, :, 1]
        input_indices = np.arange(x.shape[0])
        output_indices = np.arange(x.shape[0], x.shape[0] + y.shape[0])
        print(i // 2, i % 2)
        subplot = ax[i // 2, i % 2]
        subplot.plot(input_indices, x, label='input')
        subplot.plot(output_indices, y, label='true output')
        subplot.plot(output_indices, predicted, label='predicted output')
        subplot.legend()
        i += 1
        if i >= 4:
            break
    plt.show()


def plot_clusters(cfg):
    datagen = RawDataGenerator(cfg)
    model = LSTMRaw(cfg)
    model(next(iter(datagen.train))[0])
    model.load_weights(f'{SIMULATED_PATH}/outputs/{args.dir}/weights.h5')

    gen = NaiveVisualizationDataGenerator(cfg)
    rnn = tf.keras.layers.RNN(model.lstm_cell, return_sequences=True)
    data = []
    labels = []
    i = 0
    n_samples = 1000
    for x, _, states in gen.dev:
        states = states[:, cfg.input_width:]
        res = rnn(x)
        data += list(res.numpy().reshape((-1, res.shape[-1])))
        labels += list(states.numpy().flatten())
        i += len(labels)
        if i >= n_samples:
            break
    data = np.array(data, dtype=np.float32)[:n_samples]
    labels = np.array(labels, dtype=int)[:n_samples]
    embeddings = PCA(n_components=50).fit_transform(data)
    embeddings = TSNE(perplexity=5).fit_transform(embeddings)

    palette = sns.color_palette().as_hex()
    datasets = []
    for i, label in enumerate(label_names):
        x = embeddings[labels == i]
        dataset = {
            'label': str(label),
            'data': [{'x': float(elem[0]), 'y': float(elem[1])} for elem in x],
            'backgroundColor': palette[i]
        }
        datasets.append(dataset)
    dataset = json.dumps(dataset)
    # print(datasets)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(label_names):
        x = embeddings[labels == i]
        plt.scatter(x[:, 0], x[:, 1], label=label)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    path = f'{SIMULATED_PATH}/outputs/{args.dir}/.hydra/config.yaml'
    cfg = OmegaConf.load(path)
    # plot_clusters(cfg)
    plot_lines(cfg)
