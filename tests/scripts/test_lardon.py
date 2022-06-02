from conftest import n_examples, input_shape
import random
import numpy as np
import sys
import os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import pytest
import lardon
# TODO resolve this **!§&§é pytest problem

# @pytest.mark.parametrize("input_shape", test_shapes)
# @pytest.fixture


def test_compile(dumb_data, out_dir, dumb_callback):
    """tests compilation"""
    offline_list = lardon.compile(
        dumb_data+'/data', out_dir, valid_exts=['.npy'], callback=dumb_callback)
    assert len(offline_list) == n_examples
    assert offline_list.ndim == len(input_shape) + 1


def test_compile_evt(dumb_data, out_dir):
    with lardon.LardonParser(dumb_data+"/data", out_dir, valid_exts=['.npy'], force=True) as parser:
        for f in parser:
            data = np.load(f)
            metadata = {'dumb_meta':random.randrange(10)}
            parser.register(data, metadata, f)

def get_shapes(offline_list, batched=False):
    if batched:
        if offline_list.return_metadata:
            data, metadata = offline_list[:]
            return [(data[i].shape, metadata[i]) for i in range(len(data))]
        else:
            return [o.shape for o in offline_list[:]]
    else:
        if offline_list.return_metadata:
            return [(offline_list[i][0].shape, offline_list[i][1]) for i in range(len(offline_list))]
        else:
            return [offline_list[i].shape for i in range(len(offline_list))]


def test_parsing(dumb_data, out_dir):
    """tests data parsing"""
    offline_list = lardon.parse_folder(
        out_dir, return_metadata=False, return_indices=False)
    get_shapes(offline_list)
    get_shapes(offline_list, batched=True)



@pytest.mark.parsing
def test_metadata_parsing(dumb_data, out_dir):
    """tests metadata parsing"""
    offline_list = lardon.parse_folder(
        out_dir, return_metadata=True, return_indices=False)
    get_shapes(offline_list)
    get_shapes(offline_list, batched=True)


@pytest.mark.parsing
def test_indices_parsing(out_dir):
    offline_list = lardon.parse_folder(
        out_dir, return_metadata=True, return_indices=True)
    _, metadata = offline_list[:]
    assert "idx" in metadata[0].keys()
    get_shapes(offline_list)
    get_shapes(offline_list, batched=True)

# INDEXING TESTS


@pytest.mark.indexing
def test_one_int_indexing(dumb_data, offline_list):
    """checks indexing on first dimension"""
    x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
    idx = 1
    for i in range(len(input_shape)):
        if i < 2:
            continue
        current_slice = [slice(None, None, None)] * i + [idx]
        print('-- testing slice %s...' % current_slice)
        out = offline_list.__getitem__((0,) + tuple(current_slice))
        x_ref = x.__getitem__(tuple(current_slice))
        try:
            assert (x_ref == out).all()
        except AssertionError as e:
            offline_list.__getitem__((0,) + tuple(current_slice))
            raise e


@pytest.mark.indexing
def test_two_int_indexing(dumb_data, offline_list):
    """checks indexing on second dimension"""
    x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
    idx = 2
    # two int items
    for i in range(len(input_shape)-1):
        current_slice = [slice(None, None, None)] * i + [idx, idx]
        print('-- testing slice %s...' % current_slice)
        out = offline_list.__getitem__((0,) + tuple(current_slice))
        x_ref = x.__getitem__(tuple(current_slice))
        try:
            assert (x_ref == out).all()
        except AssertionError as e:
            offline_list.__getitem__((0,) + tuple(current_slice))
            raise e


@pytest.mark.indexing
def test_slice_indexing(dumb_data, offline_list):
    """checks slice indexing"""
    x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
    for i in range(len(input_shape)):
        current_slice = slice(0, 2)
        current_slice = [slice(None, None, None)] * i + [current_slice]
        print('-- testing slice %s...' % current_slice)
        out = offline_list.__getitem__((0,) + tuple(current_slice))
        x_ref = x.__getitem__(tuple(current_slice))
        try:
            assert (x_ref == out).all()
        except AssertionError as e:
            offline_list.__getitem__((0,) + tuple(current_slice))
            raise e


@pytest.mark.indexing
def test_random_indexing(dumb_data, offline_list):
    """checks randslice indexing"""
    x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
    length = 2
    for i in range(len(input_shape)):
        current_slice = [slice(None, None, None)] * i + \
            [lardon.randslice(length)]
        print('-- testing slice %s...' % current_slice)
        out, meta = offline_list.__getitem__((0,) + tuple(current_slice), return_metadata=True, return_indices=True)
        indices = meta['idx']
        x_ref = x.__getitem__(indices)
        try:
            assert (x_ref == out).all()
        except AssertionError as e:
            offline_list.__getitem__((0,) + tuple(current_slice))
            raise e

@pytest.mark.misc
def test_scatter(dumb_data, offline_list):
    for entry in offline_list.entries:
        shape = entry.shape
        for i in range(len(shape)):
            scattered_entry = entry.scatter(i)
            assert len(scattered_entry) == entry.shape[i]
    print('hello')
        

@pytest.mark.misc
def test_expand_dim(dumb_data, offline_list):
    x = np.expand_dims(np.reshape(
        np.arange(np.prod(input_shape)), input_shape), axis=2)
    offline_list.expand_dims(2)
    data = offline_list[0]
    assert (x == data).all()

