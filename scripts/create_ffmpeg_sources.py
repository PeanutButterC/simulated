import pandas as pd
from paths import SIMULATED_PATH, SIMULATED_DATA_ROOT
import os
from tqdm import tqdm


def create_symlinks(action_boundaries):
    for idx, row in tqdm(action_boundaries.iterrows()):
        id = row['id']
        src_dir = f'{SIMULATED_DATA_ROOT}_raw/{id}/images/CamOrthoSW'
        dst_dir = f'{SIMULATED_PATH}/crowdsourcing/images/{idx}'
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for i, frame in enumerate(range(row['start_frame'], row['end_frame'] + 1)):
            frame_padded = str(frame).zfill(5)
            dst_name = str(i).zfill(2)
            src = f'{src_dir}/{frame_padded}.png'
            dst = f'{dst_dir}/{dst_name}.png'
            os.symlink(src, dst)


if __name__ == '__main__':
    action_boundaries = pd.read_json(f'{SIMULATED_PATH}/crowdsourcing/action_boundaries.json', orient='index')
    create_symlinks(action_boundaries)
