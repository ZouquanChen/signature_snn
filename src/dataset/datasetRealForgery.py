"""Real-only features and episodes shared by training/evaluation; no SLN imports.

Keeps the legacy 12 dynamic feature definitions. Copies trajectories before
filtering, protects constant channels and supports short sequences. No on-disk
cache, augmentation, synthesis, resampling or identity renumbering is performed.
"""

from types import SimpleNamespace

import numpy as np
from scipy import signal


_FILTER_B, _FILTER_A = signal.butter(3, 15.0 / 50.0, btype="low")


def _diff(value):
    result = np.convolve(value, [0.5, 0, -0.5], mode="same")
    # np.convolve(mode='same') returns max(input, kernel) for L=2.
    if len(value) == 2:
        return np.full(2, (value[1] - value[0]) * 0.5)
    result[0], result[-1] = result[1], result[-2]
    return result


def extract_features(trajectory):
    value = np.array(trajectory, dtype=np.float64, copy=True)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError("real trajectory must be finite [L>=2,3]")
    padlen = min(3 * max(len(_FILTER_A), len(_FILTER_B)), len(value) - 1)
    value[:, :2] = signal.filtfilt(_FILTER_B, _FILTER_A, value[:, :2], axis=0, padlen=padlen)
    dx, dy = _diff(value[:, 0]), _diff(value[:, 1])
    speed = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    dv = _diff(speed)
    angle_diff = np.zeros_like(theta)
    if len(theta) > 2:
        angle_diff[1:-1] = theta[2:] - theta[:-2]
        angle_diff[0], angle_diff[-1] = angle_diff[1], angle_diff[-2]
    else:
        angle_diff[:] = theta[1] - theta[0]
    outside = np.abs(angle_diff) > np.pi
    angle_diff[outside] -= np.sign(angle_diff[outside]) * 2 * np.pi
    dtheta = np.abs(angle_diff * 0.5)
    radius = np.log((speed + 0.05) / (dtheta + 0.05))
    tangential = np.abs(speed * dtheta)
    acceleration = np.hypot(dv, tangential)
    features = np.stack((dx, dy, speed, np.cos(theta), np.sin(theta), theta,
                         radius, acceleration, dv, tangential, dtheta, value[:, 2]), axis=1)
    features = features.astype(np.float32)
    scale = features.std(axis=0)
    features = (features - features.mean(axis=0)) / np.maximum(scale, 1e-6)
    if not np.isfinite(features).all():
        raise ValueError("non-finite real-signature features")
    return features.astype(np.float32)


class RealTrainingDataset:
    featDim = 12

    def __init__(self, sigDict, taskSize=4, taskNumGen=5, taskNumNeg=10):
        self.trainKeys = sorted(sigDict)
        self.taskSize, self.taskNumGen, self.taskNumNeg = taskSize, taskNumGen, taskNumNeg
        if min(taskSize, taskNumGen, taskNumNeg) < 1 or len(sigDict) < taskSize:
            raise ValueError("positive episode sizes and enough distinct writers are required")
        self.feats, self.labels, self.is_genuine, self.writer_indices = [], [], [], []
        for label, writer in enumerate(self.trainKeys):
            samples = sigDict[writer]
            if len(samples[True]) < 1 + taskNumGen or len(samples[False]) < taskNumNeg:
                raise ValueError("not enough genuine/real forgery samples for sampling without replacement")
            groups = {}
            for genuine in (True, False):
                start = len(self.feats)
                self.feats.extend(extract_features(path) for path in samples[genuine])
                count = len(samples[genuine])
                self.labels.extend([label] * count)
                self.is_genuine.extend([genuine] * count)
                groups[genuine] = np.arange(start, start + count)
            self.writer_indices.append(groups)

    def __len__(self):
        # The shared trainer uses len(dataset) as the number of writer classes.
        return len(self.trainKeys)

    def __getitem__(self, index):
        feature = self.feats[index]
        return feature, len(feature), self.labels[index]


class TrainingBatchSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        data = self.dataset
        for _ in range(len(data)):
            writers = np.random.choice(len(data), data.taskSize, replace=False)
            batch = []
            for writer in writers:
                groups = data.writer_indices[writer]
                # Anchor + genuine positives + SAME-WRITER real skilled forgeries.
                batch.extend(np.random.choice(groups[True], 1 + data.taskNumGen, replace=False))
                batch.extend(np.random.choice(groups[False], data.taskNumNeg, replace=False))
            yield np.asarray(batch, dtype=np.int64)


class RealEvaluationDataset:
    featDim = 12

    def __init__(self, sigDict):
        self.testKeys = sorted(sigDict)
        if not self.testKeys:
            raise ValueError("empty evaluation split")
        first = sigDict[self.testKeys[0]]
        self.numGen = len(first[True])
        self.accumNumNeg = self.numGen + len(first[False])
        self.feats = []
        for writer in self.testKeys:
            samples = sigDict[writer]
            if (len(samples[True]) != self.numGen
                    or len(samples[True]) + len(samples[False]) != self.accumNumNeg):
                raise ValueError("the common verifier requires uniform per-writer sample counts")
            self.feats.extend(extract_features(path) for flag in (True, False) for path in samples[flag])

    def __len__(self):
        return len(self.testKeys)

    def __getitem__(self, index):
        feature = self.feats[index]
        return feature, len(feature), index // self.accumNumNeg


class EvaluationBatchSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        count = self.dataset.accumNumNeg
        for writer in range(len(self.dataset)):
            yield np.arange(writer * count, (writer + 1) * count)


def collate_fn(batch):
    lengths = np.asarray([item[1] for item in batch], dtype=np.int64)
    # Avoid ambiguity when both the time and feature dimensions equal 12.
    maximum = max(int(lengths.max()), 13)
    padded = np.zeros((len(batch), maximum, 12), dtype=np.float32)
    for index, (feature, length, _) in enumerate(batch):
        padded[index, :length] = feature
    return padded, lengths, np.asarray([item[2] for item in batch], dtype=np.int64)


def evaluation_module():
    return SimpleNamespace(dataset=RealEvaluationDataset, batchSampler=EvaluationBatchSampler,
                           collate_fn=collate_fn)
