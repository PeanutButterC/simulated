from paths import SIMULATED_PATH, SIMULATED_DATA_ROOT
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf
import random
import h5py


SESSION_SIZE = 10801
DATA_COLS = ['target_posX', 'target_posY', 'target_posZ', 'target_rotX', 'target_rotY', 'target_rotZ', 'target_rotW', 'hand_posX', 'hand_posY', 'hand_posZ']
ACTIONS = ['hit', 'slap', 'carry', 'turn', 'roll', 'spin', 'bounce', 'fall', 'drop', 'fall off', 'fall over', 'bump', 'slide', 'pick up', 'put down', 'push',
           'flip', 'start', 'stop', 'toss', 'throw', 'topple', 'tip', 'tumble']


class NaiveDataGenerator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cols = ['target_posX', 'target_posY', 'target_posZ',
                          'target_rotX', 'target_rotY', 'target_rotZ', 'target_rotW',
                          'hand_posX', 'hand_posY', 'hand_posZ']
        self.output_shapes = (
            (None, cfg.model.input_width, 10),
            (None, cfg.model.output_width, 10)
        )
        self.output_types = (tf.float32, tf.float32)
        self.stores = {f: f'{SIMULATED_DATA_ROOT}/sequences_{f}.h5' for f in ['train', 'dev', 'test']}

    def generate_train(self):
        return self.generate('train')

    def generate_dev(self):
        return self.generate('dev')

    def generate_test(self):
        return self.generate('test')

    @property
    def train(self):
        return tf.data.Dataset.from_generator(
            self.generate_train, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def dev(self):
        return tf.data.Dataset.from_generator(
            self.generate_dev, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def test(self):
        return tf.data.Dataset.from_generator(
            self.generate_test, output_types=self.output_types, output_shapes=self.output_shapes)

    def generate(self, type):
        with h5py.File(self.stores[type], 'r') as f:
            dset = f['df']
            indices = list(range(len(dset)))
            random.shuffle(indices)
            for idx in indices:
                data = dset[idx, :, :-1]
                ds = self.make_window_dataset(data)
                for x, y in ds:
                    yield x, y

    def make_window_dataset(self, data):
        total_width = self.cfg.model.input_width + self.cfg.model.output_width
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=total_width,
            sequence_stride=self.cfg.model.stride,
            sampling_rate=self.cfg.model.sampling_rate,
            shuffle=True,
            batch_size=self.cfg.model.batch_size,
            end_index=data.shape[0] - total_width
        )
        ds = ds.map(self.split_window)
        return ds

    def split_window(self, window):
        x = window[:, :self.cfg.model.input_width, :]
        y = window[:, self.cfg.model.input_width:, :]
        x.set_shape([None, self.cfg.model.input_width, 10])
        y.set_shape([None, self.cfg.model.output_width, 10])
        return x, y


class CPCDataGenerator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.total_sequence_length = self.cfg.model.input_width * self.cfg.model.chunk_stride + self.cfg.model.chunk_size
        self.output_shapes = {
            'o_i': (None, self.cfg.model.input_width, cfg.model.chunk_size, 10),
            'o_j': (None, cfg.model.chunk_size, 10)
        }
        self.output_types = {
            'o_i': tf.float32,
            'o_j': tf.float32
        }
        self.stores = {f: f'{SIMULATED_DATA_ROOT}/sequences_{f}.h5' for f in ['train', 'dev', 'test']}

    def generate_train(self):
        return self.generate('train')

    def generate_dev(self):
        return self.generate('dev')

    def generate_test(self):
        return self.generate('test')

    @property
    def train(self):
        return tf.data.Dataset.from_generator(self.generate_train, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def dev(self):
        return tf.data.Dataset.from_generator(self.generate_dev, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def test(self):
        return tf.data.Dataset.from_generator(self.generate_test, output_types=self.output_types, output_shapes=self.output_shapes)

    def generate(self, type):
        with h5py.File(self.stores[type], 'r') as f:
            dset = f['df']
            sequences_per_session = (SESSION_SIZE - self.total_sequence_length) // self.cfg.model.chunk_stride + 1
            indices = list(range(len(dset) * sequences_per_session))
            random.shuffle(indices)
            batch = []
            for idx in indices:
                session_idx = idx // sequences_per_session
                sequence_idx = (idx % sequences_per_session) * self.cfg.model.chunk_stride
                data = dset[session_idx, sequence_idx:(sequence_idx+self.total_sequence_length), :-1]
                batch.append(data)
                if len(batch) == self.cfg.model.batch_size:
                    batch = np.array(batch)
                    o_i, o_j = self.split_chunks(batch)
                    yield {'o_i': o_i, 'o_j': o_j}
                    batch = []

    def split_chunks(self, batch):
        chunks = []
        for i in range(0, batch.shape[1] - self.cfg.model.chunk_size + 1, self.cfg.model.chunk_stride):
            window = batch[:, i:i+self.cfg.model.chunk_size]
            chunks.append(window)
        batch = np.stack(chunks, axis=1)
        batch = tf.convert_to_tensor(batch, dtype=tf.float32)
        o_i = batch[:, :-1]
        o_j = batch[:, -1]
        o_i.set_shape([None, self.cfg.model.input_width, self.cfg.model.chunk_size, 10])
        o_j.set_shape([None, self.cfg.model.chunk_size, 10])
        return o_i, o_j


class NaiveProbeDataGenerator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cols = ['target_posX', 'target_posY', 'target_posZ',
                          'target_rotX', 'target_rotY', 'target_rotZ', 'target_rotW',
                          'hand_posX', 'hand_posY', 'hand_posZ', 'hand_state']
        self.output_shapes = (
            (None, cfg.model.input_width, 10),
            (None, cfg.model.input_width)
        )
        self.output_types = (tf.float32, tf.int64)
        self.stores = {
            'train': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/train.h5'),
            'dev': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/dev.h5'),
            'test': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/test.h5')
        }
        self.sessions = {
            'train': self.stores['train'].select_column('df', 'id').unique().tolist(),
            'dev': self.stores['dev'].select_column('df', 'id').unique().tolist(),
            'test': self.stores['test'].select_column('df', 'id').unique().tolist()
        }

    def generate_train(self):
        return self.generate('train')

    def generate_dev(self):
        return self.generate('dev')

    def generate_test(self):
        return self.generate('test')

    @property
    def train(self):
        return tf.data.Dataset.from_generator(self.generate_train, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def dev(self):
        return tf.data.Dataset.from_generator(self.generate_dev, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def test(self):
        return tf.data.Dataset.from_generator(self.generate_test, output_types=self.output_types, output_shapes=self.output_shapes)

    def generate(self, type):
        store = self.stores[type]
        sessions = self.sessions[type]
        random.shuffle(sessions)
        for session in sessions:
            df = store.select('df', 'id=%r' % session)
            assert df.index.values[0][0] == df.index.values[-1][0]
            data = df[self.data_cols].to_numpy(dtype=np.float32)
            ds = self.make_window_dataset(data)
            for x, y in ds:
                yield x, y

    def make_window_dataset(self, data):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.cfg.model.input_width,
            sequence_stride=self.cfg.model.stride,
            sampling_rate=self.cfg.model.sampling_rate,
            shuffle=True,
            batch_size=self.cfg.model.batch_size,
            end_index=data.shape[0] - self.cfg.model.input_width
        )
        ds = ds.map(self.split_window)
        return ds

    def split_window(self, window):
        states = window[:, :, -1]
        window = window[:, :, :-1]
        states.set_shape([None, self.cfg.model.input_width])
        window.set_shape([None, self.cfg.model.input_width, 10])
        return window, states


class CPCProbeDataGenerator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cols = ['target_posX', 'target_posY', 'target_posZ',
                          'target_rotX', 'target_rotY', 'target_rotZ', 'target_rotW',
                          'hand_posX', 'hand_posY', 'hand_posZ', 'hand_state']
        self.sequence_length = 90
        self.input_width = self.sequence_length // cfg.model.chunk_stride
        self.output_shapes = (
            (None, self.input_width, cfg.model.chunk_size, 10),
            (None, self.input_width)
        )
        self.output_types = (tf.float32, tf.int64)
        fname = f'sequences_sequence-length=120_sequence-stride={cfg.model.sequence_stride}'
        self.sequence_stores = {f: f'{SIMULATED_DATA_ROOT}/{fname}_{f}.h5' for f in ['train', 'dev', 'test']}

    def generate_train(self):
        return self.generate('train')

    def generate_dev(self):
        return self.generate('dev')

    def generate_test(self):
        return self.generate('test')

    @property
    def train(self):
        return tf.data.Dataset.from_generator(self.generate_train, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def dev(self):
        return tf.data.Dataset.from_generator(self.generate_dev, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def test(self):
        return tf.data.Dataset.from_generator(self.generate_test, output_types=self.output_types, output_shapes=self.output_shapes)

    def generate(self, type):
        with h5py.File(self.sequence_stores[type], 'r') as f:
            sequences = f['df']
            hand_states = f['hand_states']
            batch_size = self.cfg.model.batch_size
            indices = list(range(len(sequences)))
            random.shuffle(indices)
            indices = np.array(indices)
            for batch_idx in range(0, len(sequences) - batch_size + 1, batch_size):
                batch_indices = indices[batch_idx:batch_idx+batch_size]
                x_batch = sequences[sorted(batch_indices)][:, :-1]
                y_batch = hand_states[sorted(batch_indices)][:, :-1]
                x_batch, y_batch = self.split_chunks(x_batch, y_batch)
                yield x_batch, y_batch

    def split_chunks(self, x_batch, y_batch):
        x_chunks = []
        y_labels = []
        for i in range(0, x_batch.shape[1] - self.cfg.model.chunk_size + 1, self.cfg.model.chunk_stride):
            x_window = x_batch[:, i:i+self.cfg.model.chunk_size]
            x_chunks.append(x_window)
            y_window = y_batch[:, i:i+self.cfg.model.chunk_size]
            label_row = []
            for row in y_window:
                label = np.argmax(np.bincount(row))
                label_row.append(label)
            label_row = np.array(label_row)
            y_labels.append(label_row)
        x = np.stack(x_chunks, axis=1)
        y = np.stack(y_labels, axis=1)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.int64)
        x.set_shape([None, self.input_width, self.cfg.model.chunk_size, 10])
        y.set_shape([None, self.input_width])
        return x, y


class NaiveCrowdsourcingDataGenerator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_shapes = (
            (None, 90, 10),
            (None, len(ACTIONS))
        )
        self.output_types = (tf.float32, tf.int64)
        self.stores = {
            'train': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/train.h5'),
            'dev': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/dev.h5'),
            'test': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/test.h5')
        }
        self.sessions = {
            'train': self.stores['train'].select_column('df', 'id').unique().tolist(),
            'dev': self.stores['dev'].select_column('df', 'id').unique().tolist(),
            'test': self.stores['test'].select_column('df', 'id').unique().tolist()
        }

    def generate_train(self):
        return self.generate('train')

    def generate_dev(self):
        return self.generate('dev')

    def generate_test(self):
        return self.generate('test')

    @property
    def train(self):
        return tf.data.Dataset.from_generator(self.generate_train, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def dev(self):
        return tf.data.Dataset.from_generator(self.generate_dev, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def test(self):
        return tf.data.Dataset.from_generator(self.generate_test, output_types=self.output_types, output_shapes=self.output_shapes)

    def generate(self, type):
        crowdsourcing_df = pd.read_json(f'{SIMULATED_PATH}/crowdsourcing/production_1/response_dict_{type}.json', orient='index')
        store = self.stores[type]
        sessions = self.sessions[type]
        random.shuffle(sessions)
        batch = np.zeros((self.cfg.model.batch_size, 90, 10))
        labels = np.zeros((self.cfg.model.batch_size, len(ACTIONS)))
        idx = 0
        for session in sessions:
            df = store.select('df', 'id=%r' % session)
            assert df.index.values[0][0] == df.index.values[-1][0]
            crowdsourcing_clips = crowdsourcing_df[crowdsourcing_df['session'] == df.index.values[0][0]]
            for _, clip in crowdsourcing_clips.iterrows():
                indices = [(df.index.values[0][0], frame) for frame in range(clip['start_frame'], clip['end_frame'])]
                data = df.loc[indices][DATA_COLS].to_numpy(dtype=np.float32)
                batch[idx] = data
                for i, action in enumerate(ACTIONS):
                    labels[idx][i] = clip[action]
                idx += 1
                if idx == self.cfg.model.batch_size:
                    idx = 0
                    yield batch, labels
                    batch = np.zeros((self.cfg.model.batch_size, 90, 10))
                    labels = np.zeros((self.cfg.model.batch_size, len(ACTIONS)))


class CPCCrowdsourcingDataGenerator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.sequence_length = 90
        self.input_width = self.sequence_length // cfg.model.chunk_stride
        self.output_shapes = (
            (None, self.input_width, cfg.model.chunk_size, 10),
            (None, len(ACTIONS))
        )
        self.output_types = (tf.float32, tf.int64)
        self.stores = {
            'train': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/train.h5'),
            'dev': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/dev.h5'),
            'test': pd.HDFStore(f'{SIMULATED_DATA_ROOT}/test.h5')
        }
        self.sessions = {
            'train': self.stores['train'].select_column('df', 'id').unique().tolist(),
            'dev': self.stores['dev'].select_column('df', 'id').unique().tolist(),
            'test': self.stores['test'].select_column('df', 'id').unique().tolist()
        }

    def generate_train(self):
        return self.generate('train')

    def generate_dev(self):
        return self.generate('dev')

    def generate_test(self):
        return self.generate('test')

    @property
    def train(self):
        return tf.data.Dataset.from_generator(self.generate_train, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def dev(self):
        return tf.data.Dataset.from_generator(self.generate_dev, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def test(self):
        return tf.data.Dataset.from_generator(self.generate_test, output_types=self.output_types, output_shapes=self.output_shapes)

    def generate(self, type):
        crowdsourcing_df = pd.read_json(f'{SIMULATED_PATH}/crowdsourcing/production_1/response_dict_{type}.json', orient='index')
        store = self.stores[type]
        sessions = self.sessions[type]
        random.shuffle(sessions)
        batch = np.zeros((self.cfg.model.batch_size, 90, 10))
        labels = np.zeros((self.cfg.model.batch_size, len(ACTIONS)))
        idx = 0
        for session in sessions:
            df = store.select('df', 'id=%r' % session)
            assert df.index.values[0][0] == df.index.values[-1][0]
            crowdsourcing_clips = crowdsourcing_df[crowdsourcing_df['session'] == df.index.values[0][0]]
            for _, clip in crowdsourcing_clips.iterrows():
                indices = [(df.index.values[0][0], frame) for frame in range(clip['start_frame'], clip['end_frame'])]
                data = df.loc[indices][DATA_COLS].to_numpy(dtype=np.float32)
                batch[idx] = data
                for i, action in enumerate(ACTIONS):
                    labels[idx][i] = clip[action]
                idx += 1
                if idx == self.cfg.model.batch_size:
                    idx = 0
                    batch = self.split_chunks(batch)
                    yield batch, labels
                    batch = np.zeros((self.cfg.model.batch_size, 90, 10))
                    labels = np.zeros((self.cfg.model.batch_size, len(ACTIONS)))

    def split_chunks(self, batch):
        chunks = []
        for i in range(0, batch.shape[1] - self.cfg.model.chunk_size + 1, self.cfg.model.chunk_stride):
            window = batch[:, i:i+self.cfg.model.chunk_size]
            chunks.append(window)
        batch = np.stack(chunks, axis=1)
        batch = tf.convert_to_tensor(batch, dtype=tf.float32)
        batch.set_shape([None, self.input_width, self.cfg.model.chunk_size, 10])
        return batch


class NaiveVisualizationDataGenerator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_shapes = ((None, cfg.model.input_width, 10), (None, cfg.model.output_width, 10), (None, cfg.model.input_width + cfg.model.output_width))
        self.output_types = (tf.float32, tf.float32, tf.int64)
        self.stores = {f: pd.HDFStore(f'{SIMULATED_DATA_ROOT}/{f}.h5') for f in ['train', 'dev', 'test']}

    def generate_train(self):
        return self.generate('train')

    def generate_dev(self):
        return self.generate('dev')

    def generate_test(self):
        return self.generate('test')

    @property
    def train(self):
        return tf.data.Dataset.from_generator(self.generate_train, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def dev(self):
        return tf.data.Dataset.from_generator(self.generate_dev, output_types=self.output_types, output_shapes=self.output_shapes)

    @property
    def test(self):
        return tf.data.Dataset.from_generator(self.generate_test, output_types=self.output_types, output_shapes=self.output_shapes)

    def generate(self, type):
        store = self.stores[type]
        for df in store.select('df', chunksize=SESSION_SIZE):
            assert df.index.values[0][0] == df.index.values[-1][0]
            data = df[DATA_COLS].to_numpy()
            states = df['hand_state'].to_numpy()[:, np.newaxis]
            data = np.concatenate([data, states], axis=-1)
            ds = self.make_window_dataset(data)
            for x, y, states in ds:
                yield x, y, states

    def split_window(self, window):
        states = window[:, :, -1]
        window = window[:, :, :-1]
        x = window[:, :self.cfg.model.input_width, :]
        y = window[:, self.cfg.model.input_width:, :]
        x.set_shape([None, self.cfg.model.input_width, 10])
        y.set_shape([None, self.cfg.model.output_width, 10])
        states.set_shape([None, self.cfg.model.input_width + self.cfg.model.output_width])
        return x, y, states

    def make_window_dataset(self, window):
        window = np.array(window, dtype=np.float32)
        total_width = self.cfg.model.input_width + self.cfg.model.output_width
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=window,
            targets=None,
            sequence_length=total_width,
            sequence_stride=self.cfg.model.stride,
            sampling_rate=self.cfg.model.sampling_rate,
            shuffle=True,
            batch_size=self.cfg.model.batch_size,
            end_index=window.shape[0] - total_width
        )
        ds = ds.map(self.split_window)
        return ds


def visualize(gen):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(2, 2)
    for x in gen.train:
        o_i = x['o_i'][:, :, :, 0].numpy()
        o_j = x['o_j'][:, tf.newaxis, :, 0].numpy()
        data = np.concatenate([o_i, o_j], axis=1).reshape((o_j.shape[0], -1))
        print(data)
        print(data.shape)
        for i in range(4):
            ax[i // 2, i % 2].plot(np.arange(0, data.shape[1]), data[i])
        break
    plt.show()


if __name__ == '__main__':
    cfg = OmegaConf.load('conf/config.yaml')
    cfg.model = OmegaConf.load('conf/model/cpc.yaml')
    # cfg.model.batch_size = 10
    print(cfg)
    gen = CPCDataGenerator(cfg)
    for x in gen.dev:
        print(x['o_i'].shape, x['o_j'].shape)
        break
    # visualize(gen)
