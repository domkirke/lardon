import numpy as np, os, re, sys, dill
from tqdm import tqdm
sys.path.append('../')
from lardon.utils import checkdir
from lardon.async_list import *

def dumb_callback(filename):
    with open(filename, 'rb') as f:
        print(f)
        return np.load(f), {}

def save_as_memmap(file, data):
    current_memmap = np.memmap(file, data.dtype, 'w+', shape=data.shape)
    current_memmap[:] = np.ascontiguousarray(data)[:]

def compile(root_directory, target_directory, callback=dumb_callback, valid_exts=[".npy"], extension=".npy", **kwargs):
    tmp_dir = f"/tmp/{os.path.basename(target_directory)}"
    parsing_hash = {}
    file_list = []

    for root, directory, files in os.walk(root_directory):
        valid_files = list(filter(lambda f: True in [re.match(v, os.path.splitext(f)[1]) is not None for v in valid_exts], files))
        file_list.extend([f"{root}/{f}" for f in valid_files])
        file_prefix = re.sub(root_directory + '(/)?', '', root)
        if len(valid_files) != 0:
            checkdir(f"{tmp_dir}/{file_prefix}")

    for f in tqdm(file_list, desc="exporting files...", total=len(file_list)):
        file_name = re.sub(root_directory + '/', '', f)
        current_filename = f"{tmp_dir}/{os.path.splitext(file_name)[0]}{extension}"
        data, metadata = callback(f)
        save_as_memmap(f"{current_filename}", data)
        hash_key = f"{os.path.splitext(file_name)[0]}{extension}"
        parsing_hash[f"{hash_key}"] = {'shape':data.shape, 'strides':data.strides, 'dtype':data.dtype, **metadata}
        # entries.append(OfflineEntry(f"{current_filename}", dtype=data.dtype, shape=data.shape, strides=data.strides, **metadata))

    with open(f"{tmp_dir}/parsing.ldn", 'wb+') as f:
        dill.dump(parsing_hash, f)
    os.system(f'mv -f {tmp_dir}/* {target_directory}')

    entries = []
    for k, v in parsing_hash.items():
        entries.append(OfflineEntry(f"{target_directory}/{k}", **v))
    return OfflineDataList(entries)


def parse_folder(root_directory, **kwargs):
    assert os.path.isfile(f"{root_directory}/parsing.ldn"), "parsing file not found"
    with open(f"{root_directory}/parsing.ldn", "rb") as f:
        parsing = dill.load(f)
    parsing = {f"{root_directory}/{k}": v for k, v in parsing.items()}
    return OfflineDataList(parsing, **kwargs)



