import numpy as np


SESSION_SIZE = 10801
chunk_size = 90
chunk_stride = 30
input_sequence = np.random.uniform(size=(SESSION_SIZE,))
chunks = []
for i in range(0, len(input_sequence) - chunk_size + 1, chunk_stride):
    chunk = input_sequence[i:i+chunk_size]
    chunks.append(chunk)
chunks = np.array(chunks)
print(input_sequence.shape)
print(chunks.shape)
chunks_per_sequence = (SESSION_SIZE - chunk_size) // chunk_stride + 1
print(chunks_per_sequence)
session_size = (chunks_per_sequence - 1) * chunk_stride + chunk_size
print(session_size)
