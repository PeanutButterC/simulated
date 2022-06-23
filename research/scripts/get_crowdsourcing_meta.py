from paths import SIMULATED_PATH, SIMULATED_DATA_ROOT
import pandas as pd
import os
from tqdm import tqdm
import requests
import random


def filter(action, clip, leaky=.1):
    hand_state = clip['hand_state'].to_numpy()
    target_state = clip['target_state'].to_numpy()

    if random.uniform(0, 1) < leaky:
        return True

    # for actions where the object can't be carried the whole time, require OnCounter or OnGround
    if action in ['hit', 'slap', 'roll', 'bounce', 'fall', 'drop', 'fall off', 'fall over', 'bump', 'slide', 'pick up', 'put down', 'push', 'toss', 'throw', 'topple', 'tip', 'tumble']:
        if 0 not in target_state and 1 not in target_state:
            return False

    # for actions where contact must be made with the object but it's not held afterward, require Hit or Push
    if action in ['hit', 'slap', 'bump', 'push']:
        if 7 not in hand_state and 9 not in hand_state:
            return False

    # for actions where contact must be made with the object then it's held afterward, require Pick
    if action in ['pick up']:
        if 2 not in hand_state:
            return False

    # for actions where the object must be held then not held, require Held followed by OnCounter or OnGround
    if action in ['put down', 'toss', 'throw']:
        held = (target_state == 2)
        if (held[:-1] > held[1:]).sum() == 0:
            return False

    return True


def write_meta():
    stores = {f: pd.HDFStore(f'{SIMULATED_DATA_ROOT}/{f}.h5') for f in ['train', 'dev', 'test']}
    action_boundaries = pd.read_json(f'{SIMULATED_PATH}/crowdsourcing/action_boundaries.json', orient='index')
    action_queries = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/action_queries.csv')
    meta = []
    for _, action_query in tqdm(action_queries.iterrows(), total=len(action_queries)):
        for type in ['train', 'dev', 'test']:
            group = action_boundaries[action_boundaries['type'] == type]
            for id, rows in group.groupby('id'):
                df = stores[type].select('df', 'id=%r' % id).droplevel(0)
                for idx, row in rows.iterrows():
                    clip = df.loc[row['start_frame']:row['end_frame']+1]
                    if filter(action_query['action'], clip):
                        meta.append({
                            'action': action_query['action'],
                            'query': action_query['query'].capitalize(),
                            'url': f'https://plunarlabcit.services.brown.edu/crowdsourcing/videos/{idx}.mp4',
                            'type': type,
                            'id': id,
                            'start_frame': row['start_frame'],
                            'end_frame': row['end_frame']
                        })
    meta = pd.DataFrame(meta)
    meta.to_csv(f'{SIMULATED_PATH}/crowdsourcing/meta.csv', index_label='index')


def validate_meta():
    meta = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/meta.csv', index_col='index')
    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        url = row['url']
        response = requests.get(url)
        if response.status_code != 200:
            print(row)


if __name__ == '__main__':
    if not os.path.exists(f'{SIMULATED_PATH}/crowdsourcing/meta.csv'):
        write_meta()
    # validate_meta()
