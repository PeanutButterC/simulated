import os

assert 'SIMULATED_ROOT' in os.environ, 'set SIMULATED_ROOT'
SIMULATED_PATH = os.environ['SIMULATED_ROOT']

assert 'SIMULATED_DATA_ROOT' in os.environ, 'set SIMULATED_DATA_ROOT'
SIMULATED_DATA_ROOT = os.environ['SIMULATED_DATA_ROOT']
