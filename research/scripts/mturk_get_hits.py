import boto3
from paths import SIMULATED_PATH
import pandas as pd
import xml.etree.ElementTree as ET
import json
import numpy as np


def update_batch(batch, df, hits):
    for hit in hits:
        hit_id = hit['HITId']
        hit_row = df[df['hit_id'] == hit_id]
        assert len(hit_row) == 1, hit_row
        idx = hit_row.iloc[0]['batch_idx']
        batch.loc[idx, 'responses'] = hit['NumberOfAssignmentsCompleted']
    return batch


def get_responses(batch, df, client, hits):
    responses = []
    mapping = {'yes': 1, 'no': 2, 'unsure': 3}
    imapping = {v: k for k, v in mapping.items()}
    for hit in hits:
        hit_id = hit['HITId']
        hit_row = df[df['hit_id'] == hit_id]
        assert len(hit_row) == 1
        idx = hit_row.iloc[0]['batch_idx']
        batch_row = batch.loc[idx]
        res = client.list_assignments_for_hit(HITId=hit_id)
        hit_responses = []
        for assignment in res['Assignments']:
            root = ET.fromstring(assignment['Answer'])
            answers = []
            for elem in root.iter('{http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd}FreeText'):
                if elem.text is None:
                    elem.text = 'unsure'
                answers.append(elem.text)
            hit_responses.append(answers)
        meta_indices = json.loads(batch_row['indices'])
        urls = json.loads(batch_row['urls'])
        for i in range(10):
            row = {
                'meta_idx': meta_indices[i],
                'batch_idx': idx,
                'action': batch_row['action'],
                'type': batch_row['type'],
                'query': batch_row['query'],
                'url': urls[i]
            }
            for j in range(len(hit_responses)):
                row[f'response_{j+1}'] = mapping[hit_responses[j][i]]
            responses.append(row)
    responses = pd.DataFrame(responses)
    response_keys = [key for key in responses.columns.values if 'response' in key]
    responses['mode'] = responses[response_keys].mode(axis=1, numeric_only=True)[0].astype(int)
    for key in response_keys + ['mode']:
        responses[key] = responses[key].map(imapping)
    responses['agreement'] = responses.apply(lambda row: np.sum([(row[key] == row['mode']) for key in response_keys]) / len(response_keys), axis=1)
    return responses


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

    result = client.list_hits()
    hits = []
    while 'NextToken' in result:
        for hit in result['HITs']:
            if hit['HITId'] in hit_ids:
                hits.append(hit)
        result = client.list_hits(NextToken=result['NextToken'])
    batch = update_batch(batch, df, hits)
    responses = get_responses(batch, df, client, hits)
    responses.to_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}/responses.csv', index_label='index')

    batch.to_csv(batch_path, index_label='index')


if __name__ == '__main__':
    sandbox = False
    name = 'production_1'
    get_hits(name, sandbox=sandbox)
