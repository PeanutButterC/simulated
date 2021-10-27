import numpy as np
import pandas as pd
from paths import SIMULATED_DATA_ROOT
from simulated_data import SESSION_SIZE
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal
from tqdm import tqdm


class DynamicClustering:
    def __init__(self, delta):
        self.delta = delta

    def add_cluster(self, x):
        self.centers.append(x)
        self.radii.append(0.)
        self.indices.append(self.idx)
        self.p.append(x * x)
        self.n_samples.append(1.)
        self.idx += 1

    def compute_distance_to_clusters(self, x, centers):
        centers = np.array(centers)
        dists = cdist(x[np.newaxis, :], centers, metric='sqeuclidean')
        min_dist = np.amin(dists)
        min_idx = np.argmin(dists)
        return min_dist, min_idx

    def fit(self, x):
        self.centers = []
        self.radii = []
        self.indices = []
        self.p = []
        self.n_samples = []

        n_dims = x.shape[-1]
        self.idx = 0
        self.add_cluster(x[0])

        for i in range(1, x.shape[0]):
            min_dist, min_idx = self.compute_distance_to_clusters(x[i], self.centers)
            if min_dist > max(self.radii[min_idx], n_dims * self.delta):
                self.add_cluster(x[i])
            else:
                self.n_samples[min_idx] += 1
                self.centers[min_idx] += (x[i] - self.centers[min_idx]) / self.n_samples[min_idx]
                self.p[min_idx] += (x[i] * x[i] - self.p[min_idx]) / self.n_samples[min_idx]
                sigma = self.p[min_idx] - self.centers[min_idx] ** 2
                self.radii[min_idx] = np.sum(sigma)
                self.indices.append(min_idx)

        return self

    def transform(self, x):
        min_indices = []
        for i in range(x.shape[0]):
            _, min_idx = self.compute_distance_to_clusters(x[i], self.centers)
            min_indices.append(min_idx)
        return np.array(min_indices)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def compute_motion_energy(data, window_size=30, sigma=5, delta=3e-2):
    clusters = DynamicClustering(delta).fit_transform(data)
    clusters_pad = np.pad(clusters, window_size // 2, mode='edge')
    def motion_energy_function(x): return np.sum(np.abs(np.diff(x)) > 1e-6) / float(x.shape[0])
    motion_energy = np.array([motion_energy_function(clusters_pad[i:i+window_size]) for i in range(clusters.shape[0])])
    motion_energy = gaussian_filter1d(motion_energy, sigma)
    return motion_energy


def compute_action_boundaries(motion_energy, peak_distance=30, clip_length=90, min_length=72, max_length=96):
    peak_idx, _ = signal.find_peaks(motion_energy, distance=peak_distance)
    peak_width = signal.peak_widths(motion_energy, peak_idx, rel_height=1)[0].astype(int)
    action_boundaries = []
    for i in range(peak_idx.shape[0]):
        start_idx = max(0, peak_idx[i] - peak_width[i] // 2)
        end_idx = min(peak_idx[i] + peak_width[i] // 2, motion_energy.shape[0] - 1)
        if end_idx - start_idx > min_length and end_idx - start_idx < max_length:
            center = (start_idx + end_idx) // 2
            center = np.clip(center, clip_length // 2, motion_energy.shape[0] - clip_length // 2 - 1)
            action_boundaries.append((center - clip_length // 2, center + clip_length // 2))
    return action_boundaries


def create_crowdsourcing_clips():
    data_cols = ['target_posX', 'target_posY', 'target_posZ']
    stores = {f: pd.HDFStore(f'{SIMULATED_DATA_ROOT}/{f}.h5') for f in ['train', 'dev', 'test']}
    boundaries = []
    for type in ['train', 'dev', 'test']:
        if type == 'train':
            total = 800
        else:
            total = 100
        for df in tqdm(stores[type].select('df', chunksize=SESSION_SIZE), total=total):
            data = df[data_cols].to_numpy()
            motion_energy = compute_motion_energy(data)
            action_boundaries = compute_action_boundaries(motion_energy)
            id = df.index.values[0][0]
            for start, end in action_boundaries:
                x = data[start:end]
                motion = np.mean(np.var(x, axis=0))
                if motion < 1e-2:
                    continue
                boundaries.append({
                    'type': type,
                    'id': id,
                    'start_frame': df.index.values[start][1],
                    'end_frame': df.index.values[end][1]
                })
    boundaries = pd.DataFrame(boundaries)
    boundaries.to_json('crowdsourcing/action_boundaries.json', orient='index')


if __name__ == '__main__':
    create_crowdsourcing_clips()
