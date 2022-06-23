import glob
import os
import pandas as pd
from tqdm import tqdm
import sys
from paths import SIMULATED_PATH

store = pd.HDFStore(f'{SIMULATED_PATH}/large/data.h5', 'w')
invalid = open(f'{SIMULATED_PATH}/large/invalid.txt', 'w+')

dirs = glob.glob(f'{SIMULATED_PATH}/large_src/*')
dev_threshold = int(len(dirs) * .8)
test_threshold = int(len(dirs) * .9)
for i, dir in tqdm(enumerate(dirs), total=len(dirs)):
    fpath = f'{dir}/frames.json'
    id = os.path.basename(dir)
    try:
        df = pd.read_json(fpath, lines=True)
        if len(df[df['target_posY'] < 0]) > 0:
            invalid.write(f'{id}\n')
            continue
        if i < dev_threshold:
            df['type'] = 'train'
        elif i < test_threshold:
            df['type'] = 'dev'
        else:
            df['type'] = 'test'
        df['id'] = id
        df.set_index(['id', 'frame'], drop=True, inplace=True)
        store.append('df', df)
    except Exception:
        print(sys.exc_info())
        invalid.write(f'{id}\n')

invalid.close()
