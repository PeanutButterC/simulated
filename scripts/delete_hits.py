import boto3
from datetime import datetime


session = boto3.Session(profile_name='mturk')
client = session.client(
    service_name='mturk',
    region_name='us-east-1',
    endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
)

for item in client.list_hits(MaxResults=100)['HITs']:
    hit_id = item['HITId']
    status = client.get_hit(HITId=hit_id)['HIT']['HITStatus']
    print(status)
    if status == 'Assignable':
        response = client.update_expiration_for_hit(HITId=hit_id, ExpireAt=datetime(2015, 1, 1))
    try:
        client.delete_hit(HITId=hit_id)
    except Exception as e:
        print('Not deleted: ' + str(e))
    else:
        print('Deleted')
