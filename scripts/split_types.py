from tqdm import tqdm
import pandas as pd

store = pd.HDFStore('data.h5')
stores = {
    'train': pd.HDFStore('train.h5', 'w'),
    'dev': pd.HDFStore('dev.h5', 'w'),
    'test': pd.HDFStore('test.h5', 'w')
}
for df in tqdm(store.select('df', chunksize=10801), total=store.get_storer('df').nrows // 10801):
    assert df.index.values[0][1] == 1 and df.index.values[-1][1] == 10801
    stores[df.iloc[0]['type']].append('df', df.drop('type', axis=1))
