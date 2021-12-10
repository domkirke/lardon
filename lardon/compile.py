import numpy as np, os, re, sys, dill
from tqdm import tqdm
from collections import OrderedDict
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
    file_list = kwargs.get('file_list')

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
        data = np.ascontiguousarray(data)
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

def erase(path):
    if os.path.isdir(path):
        for r,dirs,files in os.walk(path):
            for f in files:
                os.remove(f'{r}/{f}')
            for d in dirs:
                erase(f'{r}/{d}')
        os.rmdir(path)
    else:
        os.remove(path)


def parse_folder(root_directory, drop_metadata=False, files=None, **kwargs):
    assert os.path.isfile(f"{root_directory}/parsing.ldn"), "parsing file not found"
    with open(f"{root_directory}/parsing.ldn", "rb") as f:
        parsing = dill.load(f)
    new_parsing = OrderedDict()
    metadata = OrderedDict()
    if files is not None:
        ordered_parsing = OrderedDict()
        for f in files:
            ordered_parsing[f] = parsing[f]
        parsing = ordered_parsing
    for k, v in parsing.items():
        new_parsing[f"{root_directory}/{k}"] = v
        if drop_metadata:
            new_metadata = {k: v[k] for k in filter(lambda x: x not in ['shape', 'strides', 'dtype'], list(v.keys()))}
            for k in new_metadata.keys():
                if metadata.get(k) is None:
                    metadata[k] = [new_metadata[k]]
                else:
                    metadata[k].append(new_metadata[k])
    if drop_metadata:
        return OfflineDataList(new_parsing, **kwargs), metadata
    else:
        return OfflineDataList(new_parsing, **kwargs)


# CONTEXT MANAGER FOR ONLINE WRITING
class LardonParser(object):
    def __init__(self, root_directory=None, target_directory=None, callback=None, extension=".npy", force=False):
        self.root_directory = root_directory
        self.target_directory = target_directory
        self.callback = callback
        self.extension = extension
        self.parsing_hash = {}
        self._intern_count = 0
        self.shapes = []
        self.force = force

    def __enter__(self):
        # if not os.path.isdir(self.root_directory):
        #     raise FileNotFoundError('root directory %s not found'%self.root_directory)
        if os.path.isdir(self.target_directory):
            answer = None
            if not self.force:
                while answer not in ["y", "n"]:
                    print('target directory %s is not empty. Proceed? [y/n] : '%self.target_directory)
                    answer = input()
            else:
                answer = "y"
            if answer == "y":
                erase(self.target_directory)
                os.makedirs(self.target_directory)
            else:
                raise FileExistsError()
        else:
            os.makedirs(self.target_directory)
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        # full_shape = self.get_full_shape(self.shapes)
        # self.parsing_hash['shapes'] = full_shape
        with open(f"{self.target_directory}/parsing.ldn", 'wb+') as f:
            dill.dump(self.parsing_hash, f)

    def register(self, data, metadata, filename=None):
        if filename:
            if self.root_directory is None:
                target_filename = os.path.splitext(self.target_directory+"/"+filename)[0]+self.extension
            else:
                target_filename = os.path.splitext(re.sub(self.root_directory, self.target_directory, filename))[0]+self.extension
        else:
            target_filename = f"{self.target_directory}/data_{self._intern_count}.npy"
        data = np.ascontiguousarray(data)
        current_shape = data.shape
        checkdir(os.path.dirname(target_filename))
        save_as_memmap(target_filename, data)
        current_hash = re.sub(self.target_directory+'/?', '', target_filename)
        self.parsing_hash[current_hash] ={'shape':data.shape, 'strides':data.strides, 'dtype':data.dtype, **metadata}
        self.shapes.append(current_shape)


def parse_list(arrays, file_list, target_directory, extension=".npy", metadata={}):
    tmp_dir = f"/tmp/{os.path.basename(target_directory)}"
    parsing_hash = {}
    checkdir(tmp_dir)
    checkdir(target_directory)
    for i, f in tqdm(enumerate(file_list), desc="exporting files...", total=len(file_list)):
        current_filename = f"{tmp_dir}/{os.path.splitext(f)[0]}{extension}"
        checkdir(os.path.dirname(current_filename))
        data = arrays[i]
        current_metadata = metadata[i]
        save_as_memmap(f"{current_filename}", data)
        hash_key = f"{os.path.splitext(f)[0]}{extension}"
        parsing_hash[f"{hash_key}"] = {'shape':data.shape, 'strides':data.strides, 'dtype':data.dtype, **current_metadata}

    with open(f"{tmp_dir}/parsing.ldn", 'wb+') as f:
        dill.dump(parsing_hash, f)
    os.system(f'mv -f {tmp_dir}/* {target_directory}')

    entries = []
    for k, v in parsing_hash.items():
        entries.append(OfflineEntry(f"{target_directory}/{k}", **v))
    return OfflineDataList(entries)




