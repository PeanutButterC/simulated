import boto3
from paths import SIMULATED_PATH
import pandas as pd
import json
import os
import numpy as np


def submit_batch(name, budget, cost_per_assignment, assignments_per_hit, sandbox=True):
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

    res = client.list_qualification_types(MustBeRequestable=True, MustBeOwnedByCaller=True)
    qualifications = [q['Name'] for q in res['QualificationTypes']]
    if 'Qualification Test' not in qualifications:
        qualification_questions = open(f'{SIMULATED_PATH}/crowdsourcing/qualification_questions.xml', 'r').read()
        qualification_answers = open(f'{SIMULATED_PATH}/crowdsourcing/qualification_answers.xml', 'r').read()
        qualification_attributes = {
            'Name': 'Qualification Test',
            'Description': 'Successfully completed the qualification test',
            'QualificationTypeStatus': 'Active',
            'Test': qualification_questions,
            'AnswerKey': qualification_answers,
            'TestDurationInSeconds': 5 * 60
        }
        response = client.create_qualification_type(**qualification_attributes)
        qualification_type_id = response['QualificationType']['QualificationTypeId']
    else:
        qualification_type_id = res['QualificationTypes'][(np.array(qualifications) == 'Qualification Test').argmax()]['QualificationTypeId']

    html_layout = open(f'{SIMULATED_PATH}/crowdsourcing/HIT.html', 'r').read()
    xml = '''
        <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
            <HTMLContent><![CDATA[{}]]></HTMLContent>
            <FrameHeight>0</FrameHeight>
        </HTMLQuestion>
    '''
    xml = xml.format(html_layout)

    hit_attributes = {
        'AutoApprovalDelayInSeconds': 3*24*60*60,
        'AssignmentDurationInSeconds': 30*60,
        'Reward': str(cost_per_assignment),
        'Title': 'Does the specified action occur in the 1s video clip?',
        'Description': 'Choose whether the specified action occurs in the 1 second video clip.',
        'Keywords': 'video, classification, action, verb, yes/no',
        'QualificationRequirements': [{
            'QualificationTypeId': qualification_type_id,
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [90]
        }]
    }
    hit_type = client.create_hit_type(**hit_attributes)
    hit_type_id = hit_type['HITTypeId']

    task_attributes = {
        'HITTypeId': hit_type_id,
        'LifetimeInSeconds': 3*24*60*60
    }

    results = []
    batch = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}/batch.csv', index_col='index')
    cost_per_hit = cost_per_assignment * assignments_per_hit
    max_total_hits = budget // cost_per_hit
    actions = batch['action'].unique().tolist()
    max_hits_per_action = max_total_hits // len(actions)
    max_hits = {
        'train': int(max_hits_per_action // 10 * 8),
        'dev': int(max_hits_per_action // 10),
        'test': int(max_hits_per_action // 10)
    }
    hits_to_submit = []
    for type in ['train', 'dev', 'test']:
        group = batch[batch['type'] == type]
        for _, rows in group.groupby('action'):
            i = 0
            for _, row in rows.iterrows():
                if row['responses'] < assignments_per_hit:
                    hits_to_submit.append(row)
                    i += 1
                else:
                    print(f'Hit already has {assignments_per_hit} responses, skipping')
                if i >= max_hits[type]:
                    break
    hits_to_submit = pd.DataFrame(hits_to_submit)
    for idx, row in hits_to_submit.iterrows():
        n_assignments = int(assignments_per_hit - row['responses'])
        response = client.create_hit_with_hit_type(**task_attributes, MaxAssignments=n_assignments, Question=xml.replace('${query}', row['query']).replace('${urls}', row['urls']))
        hit_type_id = response['HIT']['HITTypeId']
        results.append({
            'batch_idx': idx,
            'hit_id': response['HIT']['HITId']
        })

    results = pd.DataFrame(results)
    results.to_csv(f'{SIMULATED_PATH}/crowdsourcing/{name}/hits.csv', index_label='index')

    preview = mturk_environment['preview']
    print(f'Successfully submitted batch, preview at: {preview}?groupId={hit_type_id}')


def make_batch(name, max_hits_per_action=100):
    batch_dir = f'{SIMULATED_PATH}/crowdsourcing/{name}'
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    meta = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/meta.csv', index_col='index')
    actions = meta['action'].unique().tolist()
    hits_per_action = {
        'train': max_hits_per_action // 10 * 8,
        'dev': max_hits_per_action // 10,
        'test': max_hits_per_action // 10
    }
    hits = []
    for type in ['train', 'dev', 'test']:
        group = meta[meta['type'] == type]
        hit_indices = {k: [] for k in actions}
        for _, candidates in group.groupby('url'):
            for idx, row in candidates.iterrows():
                if len(hit_indices[row['action']]) < hits_per_action[type] * 10:
                    hit_indices[row['action']].append(idx)
        for action in actions:
            for i in range(0, len(hit_indices[action]) - 9, 10):
                indices = hit_indices[action][i:i+10]
                assert len(indices) == 10
                rows = meta.loc[indices]
                query = rows.iloc[0]['query']
                urls = rows['url'].values.tolist()
                hits.append({
                    'indices': json.dumps(indices),
                    'type': type,
                    'action': action,
                    'query': query,
                    'urls': json.dumps(urls),
                    'responses': 0
                })
    batch = pd.DataFrame(hits)
    batch.to_csv(f'{batch_dir}/batch.csv', index_label='index')


def compute_budget_ceiling(cost_per_assignment, assignments_per_hit, max_hits_per_action=100):
    meta = pd.read_csv(f'{SIMULATED_PATH}/crowdsourcing/meta.csv', index_col='index')
    max_hits = {
        'train': max_hits_per_action // 10 * 8,
        'dev': max_hits_per_action // 10,
        'test': max_hits_per_action // 10
    }
    n_hits = 0
    for type in ['train', 'dev', 'test']:
        group = meta[meta['type'] == type]
        for action, rows in group.groupby('action'):
            action_hits = min(len(rows) // 10, max_hits[type])
            n_hits += action_hits
            print(action, type, action_hits)
    cost_per_hit = cost_per_assignment * assignments_per_hit
    print(cost_per_hit * n_hits)


if __name__ == '__main__':
    cost_per_assignment = .01
    assignments_per_hit = 1
    max_hits_per_action = 10
    budget = 1200
    sandbox = True
    name = 'sandbox'
    compute_budget_ceiling(cost_per_assignment, assignments_per_hit, max_hits_per_action=max_hits_per_action)
    make_batch(name, max_hits_per_action=max_hits_per_action)
    submit_batch(name, budget, cost_per_assignment, assignments_per_hit, sandbox=sandbox)
