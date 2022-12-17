import pytest, os
import lardon
import random
import numpy as np

n_examples = 10
input_shape = (7,11,13)
# input_shape = (2, 44100)

@pytest.fixture
def dumb_data_path():
    out = "tests/dumb_dataset"
    return out

@pytest.fixture
def dumb_data(dumb_data_path):
    if not os.path.isdir(dumb_data_path):
        os.makedirs(dumb_data_path)
        if not os.path.isdir(f"{dumb_data_path}/data"):
            os.makedirs(f"{dumb_data_path}/data")
        for n in range(n_examples): 
            data = np.reshape(np.arange(np.prod(input_shape)), input_shape)
            np.save(f"{dumb_data_path}/data/dumb_{n}.npy", data)
    return dumb_data_path

@pytest.fixture
def out_dir(dumb_data_path):
    if not os.path.isdir(f"{dumb_data_path}/parsing"):
        os.makedirs(f"{dumb_data_path}/parsing")
    return f"{dumb_data_path}/parsing"
    
@pytest.fixture
def dumb_callback():
    def callback(filename):
        with open(filename, 'rb') as f:
            return np.load(f), {'dumb_meta':random.randrange(10)}
    return callback

@pytest.fixture 
def offline_list(out_dir):
    return lardon.parse_folder(out_dir, return_metadata=False, return_indices=False)
