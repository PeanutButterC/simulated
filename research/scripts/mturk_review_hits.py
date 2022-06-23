import boto3
from paths import SIMULATED_PATH
import pandas as pd
import xml.etree.ElementTree as ET
import json
import numpy as np


def get_hits(name, sandbox=True):
    if sandbox:
        mturk_environment = {
            'endpoint': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com',
            'preview': 'https://workersandbox.mturk.com/mturk/preview'
        }
    else:
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

    batch_path = f'{SIMULATED_PATH}/crowdsourcing/{name}/batch.csv'
    batch = pd.read_csv(batch_path, index_col='index')
    df = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}/hits.csv', index_col='index')
    hit_ids = set(df['hit_id'].values.tolist())

    responses = []
    mapping = {'yes': 1, 'no': 2, 'unsure': 3}
    result = client.list_reviewable_hits()
    while 'NextToken' in result:
        for hit in result['HITs']:
            if hit['HITId'] in hit_ids:
                hit_row = df[df['hit_id'] == hit['HITId']]
                assert len(hit_row) == 1
                idx = hit_row.iloc[0]['batch_idx']
                batch_row = batch.loc[idx]
                res = client.list_assignments_for_hit(HITId=hit['HITId'])
                hit_responses = []
                workers = []
                for assignment in res['Assignments']:
                    root = ET.fromstring(assignment['Answer'])
                    answers = []
                    for elem in root.iter('{http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd}FreeText'):
                        if elem.text is None:
                            elem.text = 'unsure'
                            print(hit['HITId'], assignment['WorkerId'])
                        answers.append(elem.text)
                    hit_responses.append(answers)
                    workers.append(assignment['WorkerId'])
                continue
                meta_indices = json.loads(batch_row['indices'])
                urls = json.loads(batch_row['urls'])
                for i in range(10):
                    for j in range(len(hit_responses)):
                        row = {
                            'meta_idx': meta_indices[i],
                            'batch_idx': idx,
                            'action': batch_row['action'],
                            'type': batch_row['type'],
                            'query': batch_row['query'],
                            'url': urls[i],
                            'worker': workers[j],
                            'response': mapping[hit_responses[j][i]]
                        }
                        responses.append(row)
        result = client.list_reviewable_hits(NextToken=result['NextToken'])
    responses = pd.DataFrame(responses)
    pd.set_option('display.max_colwidth', 200)
    for worker, rows in responses.groupby('worker'):
        worker_responses = rows['response'].to_numpy()
        if np.var(worker_responses) < 1e-2:
            print(worker, '\n', rows, '\n')


if __name__ == '__main__':
    sandbox = False
    name = 'production_1'
    get_hits(name, sandbox=sandbox)
