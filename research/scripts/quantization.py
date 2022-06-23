import numpy as np
import matplotlib.pyplot as plt


def get_meta(q):
    hsize = int(6.2 / q + 1)
    vsize = int(3. / q + 1)
    rsize = int(2. / q + 1)
    pos_voxels = hsize * hsize * vsize
    rot_voxels = rsize * rsize * rsize * rsize
    return {
        'hsize': hsize,
        'vsize': vsize,
        'rsize': rsize,
        'pos_voxels': pos_voxels,
        'rot_voxels': rot_voxels
    }


class Quantizer():
    def __init__(self, q=.1):
        self.q = q
        meta = get_meta(q)
        self.hsize = meta['hsize']
        self.rsize = meta['rsize']

    def normalize(self, x):
        x[:, [1, 8]] = np.clip(x[:, [1, 8]], 0, 3)
        x[:, [0, 2, 7, 9]] = np.clip(x[:, [0, 2, 7, 9]] + 3.1, 0, 6.2)
        x[:, 3:7] = np.clip(x[:, 3:7] + 1, 0, 2)
        return x

    def slice(self, x):
        quantized = np.zeros(x.shape, dtype=int)
        for d in range(x.shape[-1]):
            quantized[:, d] = np.round(x[:, d] / self.q).astype(int)
        return quantized

    def quantize(self, x):
        x = self.normalize(x)
        x = self.slice(x)
        target_pos = (x[:, 1] * self.hsize * self.hsize + x[:, 2] * self.hsize + x[:, 0]).astype(int)
        target_rot = (x[:, 3] * self.rsize * self.rsize * self.rsize + x[:, 4] * self.rsize * self.rsize + x[:, 5] * self.rsize + x[:, 6]).astype(int)
        hand_pos = (x[:, 8] * self.hsize * self.hsize + x[:, 7] * self.hsize + x[:, 9]).astype(int)
        return np.stack([target_pos, target_rot, hand_pos], axis=-1)

    def viz(self, seq_true, seq_q):
        seq_true = self.normalize(seq_true)
        seq_true *= (1 / self.q)
        target_pos_y = seq_q[:, 0] // (self.hsize * self.hsize)
        target_pos_z = (seq_q[:, 0] % (self.hsize * self.hsize)) // self.hsize
        target_pos_x = seq_q[:, 0] % self.hsize
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for _ in range(seq_q.shape[0]):
            ax.scatter(target_pos_x, target_pos_z, target_pos_y, marker='o')
            ax.scatter(seq_true[:, 0], seq_true[:, 2], seq_true[:, 1], marker='^')
        plt.show()
