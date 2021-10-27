import numpy as np
import pandas as pd
from paths import SIMULATED_DATA_ROOT
from tqdm import tqdm
from simulated_data import DATA_COLS, SESSION_SIZE
import h5py


def gather():
    output_stores = {f: f'{SIMULATED_DATA_ROOT}/sequences_{f}.h5' for f in ['train', 'dev', 'test']}
    print('loading stores')
    stores = {type: pd.HDFStore(f'{SIMULATED_DATA_ROOT}/{type}.h5', 'r') for type in ['train', 'dev', 'test']}

    for type in ['train', 'dev', 'test']:
        with h5py.File(output_stores[type], 'w') as f:
            nsessions = stores[type].get_storer('df').nrows // SESSION_SIZE
            output_dset = f.create_dataset('df', (nsessions, SESSION_SIZE, len(DATA_COLS) + 1), dtype=np.float32, chunks=(1, SESSION_SIZE, len(DATA_COLS) + 1))
            idx = 0
            indices = []
            for df in tqdm(stores[type].select('df', chunksize=SESSION_SIZE), total=nsessions):
                assert df.index.values[0][0] == df.index.values[-1][0]
                data = df[DATA_COLS + ['hand_state']].to_numpy()
                output_dset[idx] = data
                indices.append(df.index.values[0][0])
                idx += 1
            print(idx, nsessions)
            f.create_dataset('indices', data=np.array(indices, dtype='S'))


if __name__ == '__main__':
    gather()
