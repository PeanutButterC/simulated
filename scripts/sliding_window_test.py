import numpy as np

chunk_size = 120
chunk_stride = 90
input_sequence = np.random.uniform(size=(10801,))
chunks = []
for i in range(0, len(input_sequence) - chunk_size + 1, chunk_stride):
    chunk = input_sequence[i:i+chunk_size]
    print(i + chunk_size)
    chunks.append(chunk)
chunks = np.array(chunks)
print(input_sequence.shape)
print(chunks.shape)
print((10801 - chunk_size) // chunk_stride + 1)
