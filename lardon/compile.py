from .async_list import *
from .utils import checkdir
import numpy as np
import shutil, tempfile
import os
import re
import sys
import dill
from tqdm import tqdm
from typing import Tuple, Dict, Callable, List
from collections import OrderedDict

__all__ = ["default_callback", "compile", "parse_folder", "LardonParser"]

def default_callback(filename: str) -> Tuple[np.array, Dict[str, np.array]]:
    """default_callback

    Args:
        filename (str): path of the file to load

    Returns:
        Tuple[np.array, Dict[str: np.array]]: data and metadata
    """
    with open(filename, 'rb') as f:
        return np.load(f), {}


def save_as_memmap(file: str, data: np.array) -> None:
    """save a numpy array as memmap"""
    current_memmap = np.memmap(file, data.dtype, 'w+', shape=data.shape)
    current_memmap[:] = np.ascontiguousarray(data)[:]



def compile(root_directory: str, 
            target_directory: str, 
            callback: Callable = default_callback,
            valid_exts: List[str] = [".npy"],
            extension: str = ".npy",
            force:bool = True,
            verbose: bool = True,
            **kwargs):
    """compile is the main routine to parse a data folder into a lardon dataset. 

    Args:
        root_directory (str): path of data directory
        target_directory (str): path of target directory
        callback (str, optional): callback for loading the file. Defaults to dumb_callback.
        valid_exts (list, optional): allowed extensions for loaded files. Defaults to [".npy"].
        extension (str, optional): extension for parsed data. Defaults to ".npy".

    Returns:
        offline_list (OfflineDataList): parsed data
    """
    # data is first written in /tmp dir, in case the 
    tmp_dir = tempfile.gettempdir()+"/"+os.path.basename(target_directory)
    checkdir(tmp_dir)
    parsing_hash = {}
    file_list = kwargs.get('file_list', [])
    for root, directory, files in os.walk(root_directory):
        valid_files = list(filter(lambda f: True in [re.match(
            v, os.path.splitext(f)[1]) is not None for v in valid_exts], files))
        file_list.extend([f"{root}/{f}" for f in valid_files])
        file_prefix = re.sub(root_directory + '(/)?', '', root)
        if len(valid_files) != 0:
            checkdir(f"{tmp_dir}/{file_prefix}")

    if len(file_list) == 0:
        raise FileNotFoundError("Directory %s is empty or could not be found"%root_directory)

    # export data
    if verbose:
        iterator = tqdm(file_list, desc="exporting files...", total=len(file_list)) 
    else:
        iterator = file_list
    for f in iterator:
        file_name = re.sub(root_directory + '/', '', f)
        current_filename = f"{tmp_dir}/{os.path.splitext(file_name)[0]}{extension}"
        data, metadata = callback(f)
        data = np.ascontiguousarray(data)
        save_as_memmap(f"{current_filename}", data)
        hash_key = f"{os.path.splitext(file_name)[0]}{extension}"
        parsing_hash[f"{hash_key}"] = {'shape': data.shape,
                                       'strides': data.strides, 'dtype': data.dtype, **metadata}
        # entries.append(OfflineEntry(f"{current_filename}", dtype=data.dtype, shape=data.shape, strides=data.strides, **metadata))

    with open(f"{tmp_dir}/parsing.ldn", 'wb+') as f:
        dill.dump(parsing_hash, f)
    # move folder to correct place
    if os.path.exists(target_directory):
        answer = None
        if not force:
            raise FileExistsError
        else:
            shutil.rmtree(target_directory, ignore_errors=False)
    shutil.move(tmp_dir, target_directory)
    entries = []
    # build offline data list
    for k, v in parsing_hash.items():
        entries.append(OfflineEntry(f"{target_directory}/{k}", **v))
    return OfflineDataList(entries)


def erase(path):
    """Deletes a full folder."""
    if os.path.isdir(path):
        for r, dirs, files in os.walk(path):
            for f in files:
                os.remove(f'{r}/{f}')
            for d in dirs:
                erase(f'{r}/{d}')
        os.rmdir(path)
    else:
        os.remove(path)


def parse_folder(root_directory: str, drop_metadata: bool=False, files: List[str]=None, **kwargs) -> OfflineDataList:
    """parse_folder parses a folder exported beforehand by the lardon package.

    Args:
        root_directory (str): parsed data directory
        drop_metadata (int, optional): decides if the functions returns the metadata. Defaults to False.
        files (List[str], optional): restricts the data to a list of file. Defaults to None.

    Returns:
        offline_data_list (OfflineDataList): parsed data
    """
    assert os.path.isfile(
        f"{root_directory}/parsing.ldn"), "parsing file not found"
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
            new_metadata = {k: v[k] for k in filter(
                lambda x: x not in ['shape', 'strides', 'dtype'], list(v.keys()))}
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
    def __init__(self, root_directory=None, target_directory=None, callback=None, valid_exts=[".npy"], extension=".npy", force=False):
        """LardonParser is a environement-driven method to parse a datafolder to a lardon Dataset. Arrays are 
        registered with the register callback of the environment object, that parses it successively.

        Args:
            root_directory (str, optional): Dataset path. Defaults to None.
            target_directory (str, optional): Target directory. Defaults to None.
            callback (Callable, optional): Import callback. Defaults to None.
            extension (str, optional): Extension of exported files. Defaults to ".npy".
            force (bool, optional): Overwrites if the file already exists. Defaults to False.
        """
        self.root_directory = root_directory
        self.target_directory = target_directory
        self.callback = callback
        self.extension = extension
        self.valid_exts = valid_exts
        self.files = []
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
                    print(
                        'target directory %s is not empty. Proceed? [y/n] : ' % self.target_directory)
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

        if self.root_directory is not None:
            for root, directory, files in os.walk(self.root_directory):
                valid_files = list(filter(lambda f: True in [re.match(
                    v, os.path.splitext(f)[1]) is not None for v in self.valid_exts], files))
                self.files.extend([f"{root}/{f}" for f in valid_files])
                file_prefix = re.sub(self.root_directory + '(/)?', '', root)
                if len(valid_files) != 0:
                    checkdir(f"{self.target_directory}/{file_prefix}")
        return self

    def __iter__(self):
        return iter(self.files)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # full_shape = self.get_full_shape(self.shapes)
        # self.parsing_hash['shapes'] = full_shape
        with open(f"{self.target_directory}/parsing.ldn", 'wb+') as f:
            dill.dump(self.parsing_hash, f)

    def register(self, data, metadata, filename=None):
        """register : registers a data file in the parsed dataset.

        Args:
            data (np.array): data to write
            metadata (Dict[str, np.array]): metadata to write
            filename (str, optional): name of parsed file. Defaults to None.
        """
        if filename:
            if self.root_directory is None:
                target_filename = os.path.splitext(
                    self.target_directory+"/"+filename)[0]+self.extension
            else:
                target_filename = os.path.splitext(
                    re.sub(self.root_directory, self.target_directory, filename))[0]+self.extension
        else:
            target_filename = f"{self.target_directory}/data_{self._intern_count}.npy"
        data = np.ascontiguousarray(data)
        current_shape = data.shape
        checkdir(os.path.dirname(target_filename))
        save_as_memmap(target_filename, data)
        current_hash = re.sub(self.target_directory+'/?', '', target_filename)
        self.parsing_hash[current_hash] = {
            'shape': data.shape, 'strides': data.strides, 'dtype': data.dtype, **metadata}
        self.shapes.append(current_shape)
        self._intern_count += 1

"""
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
        parsing_hash[f"{hash_key}"] = {'shape': data.shape,
                                       'strides': data.strides, 'dtype': data.dtype, **current_metadata}

    with open(f"{tmp_dir}/parsing.ldn", 'wb+') as f:
        dill.dump(parsing_hash, f)
    os.system(f'mv -f {tmp_dir}/* {target_directory}')

    entries = []
    for k, v in parsing_hash.items():
        entries.append(OfflineEntry(f"{target_directory}/{k}", **v))
    return OfflineDataList(entries)
"""