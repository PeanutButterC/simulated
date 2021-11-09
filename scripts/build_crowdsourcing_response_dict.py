import pandas as pd
from paths import SIMULATED_PATH
from tqdm import tqdm


def build_crowdsourcing_response_dict():
    meta = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/meta.csv')
    responses = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/production_1/responses.csv')
    actions = meta['action'].unique().tolist()
    response_lookup = {'no': 0, 'yes': 1, 'unsure': 0}
    for type, rows in responses.groupby('type'):
        df = pd.DataFrame(columns=['index', 'session', 'start_frame', 'end_frame'] + actions).set_index('index', drop=True)
        for _, row in tqdm(rows.iterrows(), total=len(rows)):
            meta_row = meta.loc[row['meta_idx']]
            entries = df[(df['session'] == meta_row['id']) & (df['start_frame'] == meta_row['start_frame']) & (df['end_frame'] == meta_row['end_frame'])]
            if len(entries) == 0:
                entry = {
                    'session': meta_row['id'],
                    'start_frame': meta_row['start_frame'],
                    'end_frame': meta_row['end_frame']
                }
                for action in actions:
                    entry[action] = -1
                df = df.append(entry, ignore_index=True)
                entries = df[(df['session'] == meta_row['id']) & (df['start_frame'] == meta_row['start_frame']) & (df['end_frame'] == meta_row['end_frame'])]
            assert len(entries) == 1
            df.loc[entries.iloc[0].name][row['action']] = response_lookup[row['mode']]
        df.to_json(f'{SIMULATED_PATH}/crowdsourcing/production_1/response_dict_{type}.json', orient='index')


if __name__ == '__main__':
    build_crowdsourcing_response_dict()
