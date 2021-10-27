import pandas as pd
from paths import SIMULATED_PATH
import boto3
import matplotlib.pyplot as plt
import seaborn as sns


def worker_analysis(name):
    worker_statistics = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}worker_statistics.csv', index_col=0, squeeze=True)
    worker_statistics.sort_values(axis=0, inplace=True, ascending=False)
    print('n_workers: ' + str(len(worker_statistics)))
    print('mean:' + str(worker_statistics.mean()))
    print('median: ' + str(worker_statistics.median()))
    print('max: ' + str(worker_statistics.max()))
    print('skew: ' + str(worker_statistics.skew()))


def get_worker_statistics(name):
    hit_df = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}/hits.csv', index_col='index')
    hit_ids = set(hit_df['hit_id'].values.tolist())
    mturk_environment = {
        'endpoint': 'https://mturk-requester.us-east-1.amazonaws.com',
        'preview': 'https://www.mturk.com/mturk/preview'
    }
    session = boto3.Session(profile_name='mturk')
    client = session.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url=mturk_environment['endpoint']
    )
    worker_statistics = {}
    result = client.list_hits()
    while 'NextToken' in result:
        for hit in result['HITs']:
            if hit['HITId'] in hit_ids:
                res = client.list_assignments_for_hit(HITId=hit['HITId'])
                for assignment in res['Assignments']:
                    worker_id = assignment['WorkerId']
                    if worker_id not in worker_statistics:
                        worker_statistics[worker_id] = 0
                    worker_statistics[worker_id] += 1
        result = client.list_hits(NextToken=result['NextToken'])
    s = pd.Series(worker_statistics)
    s.to_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}worker_statistics.csv')


def response_heatmap(name):
    responses = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}/responses.csv', index_col='index')
    actions = responses['action'].unique().tolist()
    heatmap = []
    for action, group in responses[responses['mode'] == 'yes'].groupby('action'):
        for other in actions:
            overlap = responses[(responses['action'] == other) & (responses['url'].isin(group['url'].tolist()))]
            match = len(overlap[overlap['mode'] == 'yes']) / len(overlap)
            heatmap.append({
                'clips labeled': action,
                'proportion also labeled': other,
                'match': match
            })
    heatmap = pd.DataFrame(heatmap)
    heatmap = heatmap.pivot('clips labeled', 'proportion also labeled', 'match')
    row_idx = heatmap.mean().sort_values(ascending=True).index
    row_idx.name = 'clips labeled'
    col_idx = heatmap.mean().sort_values(ascending=False).index
    heatmap = heatmap.reindex(index=row_idx, columns=col_idx)
    print(heatmap)
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.heatmap(heatmap, cmap='Blues')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def agreement_bar(name):
    responses = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}/responses.csv', index_col='index')
    agreement = responses.groupby('action')['agreement'].mean().sort_values(ascending=False).reset_index()
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.barplot(x='action', y='agreement', data=agreement, color=sns.color_palette('Blues', n_colors=2)[1])
    plt.ylim([0.5, 1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    name = 'production_1'
    # get_worker_statistics(name)
    # worker_analysis(name)
    # response_heatmap(name)
    agreement_bar(name)
