import sys, random, os, numpy as np
sys.path.append('../')
import lardon, pytest

n_examples = 10

# @pytest.mark.parametrize("input_shape", test_shapes)
# @pytest.fixture

## EXPORT TESTS
input_shape = (100, 75, 18)

def callback(filename):
    with open(filename, 'rb') as f:
        return np.load(f), {'dumb_meta':random.randrange(10)}

@pytest.mark.parsing
@pytest.mark.indexing
@pytest.mark.misc
def test_export(tmpdir):
    # input_shape = list(input_shape)
    datas = []
    localdir = tmpdir.dirname
    for ex in range(n_examples):
        for i, inp in enumerate(input_shape):
            if inp is None:
                input_shape[i] = random.randrange(40)
        x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
        lardon.checkdir(f"{localdir}/data")
        lardon.checkdir(f"{localdir}/parsed")
        with open(f"{localdir}/data/dumb_{ex}.npy", 'wb') as f:
            np.save(f, x)
        datas.append(x)
    offline_list = lardon.compile(localdir+'/data', localdir+"/parsed", valid_exts=['.npy'], callback=callback)
    assert len(offline_list) == len(datas)
    assert offline_list.ndim == len(input_shape) + 1

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

@pytest.mark.parsing
def test_parsing(tmpdir):
    offline_list = lardon.parse_folder(tmpdir.dirname + '/parsed', return_metadata=False, return_indices=False)
    get_shapes(offline_list)
    get_shapes(offline_list, batched=True)

@pytest.mark.parsing
def test_metadata_parsing(tmpdir):
    offline_list = lardon.parse_folder(tmpdir.dirname + '/parsed', return_metadata=True, return_indices=False)
    get_shapes(offline_list)
    get_shapes(offline_list, batched=True)

@pytest.mark.parsing
def test_indices_parsing(tmpdir):
    offline_list = lardon.parse_folder(tmpdir.dirname + '/parsed', return_metadata=True, return_indices=True)
    _, metadata = offline_list[:]
    assert "idx" in metadata[0].keys()
    get_shapes(offline_list)
    get_shapes(offline_list, batched=True)

# INDEXING TESTS
@pytest.mark.indexing
def test_one_int_indexing(tmpdir):
    offline_list = lardon.parse_folder(tmpdir.dirname + '/parsed', return_metadata=False, return_indices=False)
    x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
    idx = 1
    for i in range(len(input_shape)):
        if i < 2:
            continue
        current_slice = [slice(None, None, None)] * i + [idx]
        print('-- testing slice %s...'%current_slice)
        out = offline_list.__getitem__((0,) + tuple(current_slice))
        x_ref = x.__getitem__(tuple(current_slice))
        try:
            assert (x_ref == out).all()
        except AssertionError as e:
            offline_list.__getitem__((0,) + tuple(current_slice))
            raise e

@pytest.mark.indexing
def test_two_int_indexing(tmpdir):
    offline_list = lardon.parse_folder(tmpdir.dirname + '/parsed', return_metadata=False, return_indices=False)
    x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
    idx = 2
    # two int items
    for i in range(len(input_shape)-1):
        current_slice = [slice(None, None, None)] * i + [idx, idx]
        print('-- testing slice %s...'%current_slice)
        out = offline_list.__getitem__((0,) + tuple(current_slice))
        x_ref = x.__getitem__(tuple(current_slice))
        try:
            assert (x_ref == out).all()
        except AssertionError as e:
            offline_list.__getitem__((0,) + tuple(current_slice))
            raise e

@pytest.mark.indexing
def test_slice_indexing(tmpdir):
    offline_list = lardon.parse_folder(tmpdir.dirname + '/parsed', return_metadata=False, return_indices=False)
    x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
    for i in range(len(input_shape)):
        current_slice = slice(0, 2)
        current_slice = [slice(None, None, None)] * i + [current_slice]
        print('-- testing slice %s...'%current_slice)
        out = offline_list.__getitem__((0,) + tuple(current_slice))
        x_ref = x.__getitem__(tuple(current_slice))
        try:
            assert (x_ref == out).all()
        except AssertionError as e:
            offline_list.__getitem__((0,) + tuple(current_slice))
            raise e


@pytest.mark.indexing
def test_random_indexing(tmpdir):
    offline_list = lardon.parse_folder(tmpdir.dirname + '/parsed', return_metadata=True, return_indices=True)
    x = np.reshape(np.arange(np.prod(input_shape)), input_shape)
    length = 2
    for i in range(len(input_shape)):
        current_slice = [slice(None, None, None)] * i + [lardon.randslice(length)]
        print('-- testing slice %s...'%current_slice)
        out, meta = offline_list.__getitem__((0,) + tuple(current_slice))
        indices = meta['idx']
        x_ref = x.__getitem__(indices)
        try:
            assert (x_ref == out).all()
        except AssertionError as e:
            offline_list.__getitem__((0,) + tuple(current_slice))
            raise e

@pytest.mark.misc
def test_expand_dim(tmpdir):
    offline_list = lardon.parse_folder(tmpdir.dirname + '/parsed', return_metadata=False, return_indices=False)
    x = np.expand_dims(np.reshape(np.arange(np.prod(input_shape)), input_shape), axis=2)
    offline_list.expand_dims(2)
    data = offline_list[0]
    assert (x == data).all()







