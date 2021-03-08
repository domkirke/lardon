import numpy as np


def as_list(data, **kwargs):
    return data

def pad(data, **kwargs):
    shapes = [d.shape for d in data]
    assert len(set([len(d) for d in shapes])) == 1, "pad batching not available with variable number of dimensions"
    target_shape = np.amax(np.array(shapes), axis=0)
    new_data = [None]*len(data)
    for i, d in enumerate(data):
        if (target_shape > np.array(d.shape)).any():
            pad_values = [(0, target_shape[i] - d.shape[i]) for i in range(len(target_shape))]
            d = np.pad(d, pad_values, **kwargs)
        new_data[i] = d
    return np.stack(new_data, axis=0)

def crop(data, **kwargs):
    shapes = [d.shape for d in data]
    assert len(set([len(d) for d in shapes])) == 1, "pad batching not available with variable number of dimensions"
    target_shape = np.amin(np.array(shapes), axis=0)
    new_data = [None]*len(data)
    for i, d in enumerate(data):
        if (target_shape < np.array(d.shape)).any():
            idxs = tuple([slice(0, t) for t in target_shape])
            d = d.__getitem__(idxs)
        new_data[i] = d
    return np.stack(new_data, axis=0)
