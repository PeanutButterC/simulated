import boto3
from tqdm import tqdm


def approve_hits(hit_type_id, sandbox=True):
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

    result = client.list_reviewable_hits(HITTypeId=hit_type_id)
    while 'NextToken' in result:
        hits = result['HITs']
        for hit in tqdm(hits):
            assignments = client.list_assignments_for_hit(HITId=hit['HITId'])['Assignments']
            for assignment in assignments:
                if assignment['AssignmentStatus'] != 'Submitted':
                    continue
                client.approve_assignment(AssignmentId=assignment['AssignmentId'])
        result = client.list_reviewable_hits(HITTypeId=hit_type_id, NextToken=result['NextToken'])


if __name__ == '__main__':
    sandbox = False
    hit_type_id = '3SUQABPVM4PXPX8HB90VK7696JIUWM'
    approve_hits(hit_type_id, sandbox=sandbox)
