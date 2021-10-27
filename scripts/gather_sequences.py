import numpy as np
import pandas as pd
from paths import SIMULATED_DATA_ROOT
from tqdm import tqdm
import argparse
from simulated_data import DATA_COLS, SESSION_SIZE
import h5py


def gather(sequence_length, sequence_stride):
    fname = f'sequences_sequence-length={sequence_length}_sequence-stride={sequence_stride}'
    output_stores = {f: f'{SIMULATED_DATA_ROOT}/{fname}_{f}.h5' for f in ['train', 'dev', 'test']}
    print('loading stores')
    stores = {f: pd.HDFStore(f'{SIMULATED_DATA_ROOT}/{f}.h5') for f in ['train', 'dev', 'test']}
    print('getting sessions')
    sessions = {
        'train': stores['train'].select_column('df', 'id').unique().tolist(),
        'dev': stores['dev'].select_column('df', 'id').unique().tolist(),
        'test': stores['test'].select_column('df', 'id').unique().tolist(),
    }

    for type in ['train', 'dev', 'test']:
        print(type)
        with h5py.File(output_stores[type], 'w') as f:
            n_sequences = len(sessions[type]) * ((SESSION_SIZE - sequence_length) // sequence_stride + 1)
            output_dset = f.create_dataset('df', (n_sequences, sequence_length, len(DATA_COLS)), dtype=np.float32)
            output_hand_states = f.create_dataset('hand_states', (n_sequences, sequence_length), dtype=np.int64)
            idx = 0
            indices = []
            for session in tqdm(sessions[type]):
                df = stores[type].select('df', 'id=%r' % session)
                assert len(df) == SESSION_SIZE
                data = df[DATA_COLS].to_numpy()
                hand_states = df['hand_state'].to_numpy(dtype=np.int64)
                for i in range(0, SESSION_SIZE - sequence_length + 1, sequence_stride):
                    sequence = data[i:i+sequence_length]
                    hand_state_seq = hand_states[i:i+sequence_length]
                    output_dset[idx] = sequence
                    output_hand_states[idx] = hand_state_seq
                    indices.append(f'{df.index.values[i][0]},{df.index.values[i][1]}')
                    idx += 1
            print(idx, n_sequences)
            f.create_dataset('indices', data=np.array(indices, dtype='S'))


if __name__ == '__main__':
    # 90 + chunk_size = sequence_length
    # sequence_length // stride = chunks per sequence
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', type=int, default=30, help='size of input chunks for the contrastive model')
    parser.add_argument('--chunk_stride', type=int, default=30, help='stride of input chunks for the contrastive model, equal to chunk_size is non-overlapping')
    parser.add_argument('--sequence_stride', type=int, default=90, help='stride for whole sequences from the dataset')
    args = parser.parse_args()
    if not (90 + args.chunk_size) % args.chunk_stride == 0:
        raise ValueError('Invalid stride')
    sequence_length = 90 + args.chunk_size
    gather(sequence_length, args.sequence_stride)
