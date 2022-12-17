import numpy as np, random, sys, numbers
sys.path.append('../')
from lardon import batch

batch_hash = {'list':batch.as_list,'pad': batch.pad, 'crop':batch.crop}


def get_full_shape(shapes):
    ndims = set([len(s) for s in shapes])
    if len(ndims) > 1:
        return None
    ndims = list(ndims)[0]
    full_shape = [None] * ndims
    for dim in range(ndims):
        current_shapes = list(set([s[dim] for s in shapes]))
        if len(current_shapes) == 1:
            full_shape[dim] = current_shapes[0]
        else:
            full_shape[dim] = (min(current_shapes), max(current_shapes))
    return full_shape


class NoShapeError(Exception):
    pass

class ShapeError(Exception):
    def __init__(self, true_shape, false_shape):
        self.true_shape = true_shape
        self.false_shape = false_shape
    def __repr__(self):
        print('ShapeError(%s != %s)'%(self.true_shape, self.false_shape))

def is_same(x):
    return len(set(x)) == 1

def get_full_idx(idx, shape):
    n_idx = len(idx)
    dim = len(shape)
    full_idx = [None]*dim
    for d in range(dim):
        if d < n_idx:
            if isinstance(idx[d], int):
                full_idx[d] = (idx[d])
            else:
                full_idx[d] = idx[d]
        else:
            full_idx[d] = slice(None, None, None)
    return full_idx

## Memmap reading routines
def get_final_dim(sl, shape, squeeze=True):
    final_dim = [None]*len(shape)
    for i, s in enumerate(sl):
        if isinstance(s, int):
            if squeeze:
                continue
            else:
                final_dim[i] = 1
        elif isinstance(s, slice):
            start = s.start or 0
            stop = s.stop or shape[i]
            step = s.step or 1
            final_dim[i] = np.round((stop - start)/step).astype(np.int32)
    final_dim = np.array(list(filter(lambda x: x is not None, final_dim)))
    return final_dim

def load_memmap(fp, idx, shape, strides, dtype, squeeze=True):
    # check if strides are sorted (mandatory)
    def remove_eqs(s):
        d = []
        for s_tmp in s:
            if s_tmp in d:
                continue
            d.append(s_tmp)
        return d
    sort = np.argsort(remove_eqs(strides))
    assert (np.unique(np.diff(sort)) == -1).all(), 'array is not contiguous'

    # init iteration
    offsets = np.array([0])
    shapes = np.array([shape])
    # detect last slicing index
    end_idx = len(idx)
    for i in reversed(idx):
        if i == slice(None):
            end_idx -= 1
        else:
            break
    # get complete index (why?)
    idx = get_full_idx(idx, shape)
    # get final array shape
    final_shape = get_final_dim(idx, shape)
    final_shape_idx = 0
    offsets = np.array([0])
    # for dim in range(end_idx):
    #     if isinstance(idx[dim], int):
    #         offsets = offsets + strides[dim] * idx[dim]
    #         shapes[dim] = 1
    #     elif isinstance(idx[dim], slice):

    for dim in range(end_idx):
        if isinstance(idx[dim], int):
            # offsets = offsets + strides[dim] * idx[dim]
            # shapes[:, dim] = 1
            if dim == 0:
                offsets = (offsets[np.newaxis] + (idx[dim] * strides[dim])).flatten()
                shapes[:, dim] = 1
            else:
                offsets = offsets[np.newaxis].repeat(shapes[0][dim - 1], 0) + (np.arange(shapes[0][dim - 1])[np.newaxis].T*strides[dim-1])
                offsets = (offsets + (idx[dim] * strides[dim])).flatten()
                shapes = shapes.repeat(shapes[0][dim - 1], 0)
                shapes[:, dim] = 1
                shapes[:, dim-1] = 1
        elif isinstance(idx[dim], slice):
            # if idx[dim].step is None:
            #     start = idx[dim].start if idx[dim].start is not None else 0
            #     offsets = offsets + strides[dim] * start
            #     shapes[:, dim] = final_shape[final_shape_idx]
            # else:
            #     assert idx[dim].step > 0, "zero or negative step is not allowed (found at dim %d)"%dim
            if dim == (end_idx - 1):
                offsets = (offsets[np.newaxis] + (np.array([idx[dim].start]) * strides[dim])[:, np.newaxis]).flatten()
                shapes[:, dim] = (idx[dim].stop - idx[dim].start)
            else:
                dim_idx = list(range(shape[dim]))[idx[dim]]
                offsets = (offsets[np.newaxis] + (np.array(dim_idx) * strides[dim])[:, np.newaxis]).flatten()
                # offsets.sort()
                shapes = shapes.repeat(len(dim_idx), 0)
                shapes[:, dim] = 1
        final_shape_idx += 1
    if len(final_shape) == 0:
        arr = np.array([0], dtype=dtype)
    else:
        arr = np.zeros(np.prod(final_shape), dtype=dtype)
    current_idx = 0
    offsets.sort()
    for i in range(len(offsets)):
        current_slice = slice(current_idx, current_idx + np.prod(shapes[i]))
        arr[current_slice] = np.array(
            np.memmap(fp, dtype=dtype, mode="r", shape=tuple(shapes[i]), offset=int(offsets[i]))).flatten()
        current_idx += np.prod(shapes[i])
    if len(final_shape) > 0:
        arr = arr.reshape(final_shape)
    return arr

class randslice(object):
    def __init__(self, length=1, start=None, stop=None, step=None):
        self.length = length
        self.start = start
        self.stop = stop
        self.step = None

    def __repr__(self):
        return ("randslice(length=%s"%self.length) + \
               (", start=%s"%self.start if self.start is not None else "") + \
               (", stop=%s"%self.stop if self.stop is not None else "") +\
               (")")

    def sample(self, shape=None):
        if self.start is None or self.stop is None:
            assert shape is not None, "under-complete randslice must be called with a valid shape"
        start = self.start or 0
        end = self.stop or shape - self.length
        if start==end:
            random_idx = 0
        else:
            random_idx = random.randrange(start, end)
        if self.length == 1:
            idx = random_idx
        else:
            idx = slice(random_idx, random_idx+self.length, self.step)
        return idx

def sample_idx(idx, shape):
    if isinstance(idx, int):
        return idx
    elif isinstance(idx, (tuple, list)):
        idx = list(idx)
        for i, ix in enumerate(idx):
            if isinstance(ix, randslice):
                idx[i] = ix.sample(shape[i])
    return tuple(idx)


## Selector objects
class Selector(object):
    """takes everything"""
    def __init__(self, **kwargs):
        """
        Basic Selector object that returns the entire file.
        :param kwargs:
        """
        pass

    def __repr__(self):
        return "Selector()"

    def __getitem__(self, *args):
        return IndexPick(args)

    def get_shape(self, shape):
        return shape

    def __call__(self, file, shape=None, dtype=np.float64, return_indices=True, **kwargs):
        data = np.array(np.memmap(file, dtype=dtype, mode='r', offset=0, shape=shape))
        if return_indices:
            return data, np.array([0])
        else:
            return data

def is_empty_slice(sl):
    if not isinstance(sl, slice):
        return False
    return (sl.start == None or sl.start is 0) and (sl.stop == None or sl.stop == -1) and (sl.step is None or sl.step == 1)


class IndexPick(Selector):
    def __init__(self, idx=None, **kwargs):
        assert idx is not None
        self.idx =  idx

    def  __repr__(self):
        if isinstance(self.idx, tuple):
            return "IndexPick%s" % (str(self.idx))
        else:
            return "IndexPick(%s)" % (self.idx)

    def __getitem__(self, *args):
        if set(map(lambda x: is_empty_slice(x), args)) != {True}:
            idx = []
            current_pos = 0
            for i in range(len(self.idx)):
                if isinstance(self.idx[i], int):
                    idx.append(self.idx[i])
                if isinstance(self.idx[i], slice):
                    if current_pos >= len(args):
                        idx.append(self.idx[i])
                    if isinstance(args[current_pos], int):
                        idx.append((self.idx[i].start or 0) + args[current_pos])
                        current_pos += 1
                    elif isinstance(args, slice):
                        sl1 = self.idx[i]; sl2 = args[current_pos]
                        if is_empty_slice(sl1) and is_empty_slice(sl2):
                            idx.append(slice(None))
                        elif is_empty_slice(sl1):
                            idx.append(sl2)
                        elif is_empty_slice(sl2):
                            idx.append(sl1)
                        start = sl1.start + sl2.start
                        if (sl1.stop is None) and (sl2 is None):
                            stop = None
                        elif sl1.stop is None:
                            stop = sl2.stop
                        elif sl2.stop is None:
                            stop = sl1.stop
                        else:
                            stop = start + sl2.stop
                        step = sl1.step * sl2.step
                        idx.append(slice(start, stop, step))
                        current_pos += 1
                    else:
                        raise NotImplementedError
            if current_pos < len(args):
                for i in range(current_pos, len(args)):
                    idx.append(args[i])
            return IndexPick(idx=tuple(idx))
        else:
            return self

    def get_shape(self, shape):
        new_shape = list(shape)
        for i, idx in enumerate(self.idx):
            if i >= len(shape):
                break
            if isinstance(idx, slice):
                new_shape[i] = len(range(shape[i])[idx])
            elif hasattr(idx, "__iter__"):
                new_shape[i] = len(idx)
            elif isinstance(idx, randslice):
                if idx.length == 1:
                    new_shape[i] = None
                else:
                    new_shape[i] = idx.length
            else:
                new_shape[i] = None
        new_shape = list(filter(lambda x: x is not None, new_shape))
        return tuple(new_shape)

    def check_idx(self, idx, shape):
        new_idx = []
        for i, c_i in enumerate(idx):
            if i > len(shape):
                break
            if isinstance(c_i, slice):
                if c_i.stop is not None:
                    assert c_i.stop < shape[i], "slice %s incompatible with shape %s at dim %d"%(c_i.stop, shape, i)
            elif isinstance(c_i, int):
                assert c_i < shape[i], "index %d incompatible with shape %s at dim %d"%(c_i, shape, i)
            elif hasattr(c_i, "__iter__"):
                max_idx = max(c_i)
                assert max_idx < shape[i], "tried to retrieve idx %s, but shape at dim %d is %s"%(max_idx, i, c_i)
            new_idx.append(c_i)
        return new_idx

    def __call__(self, file, shape, strides, dtype=np.float64, return_indices=False, **kwargs):
        idx = self.check_idx(self.idx, shape)
        idx = sample_idx(idx, shape)
        data = load_memmap(file, idx, shape, strides, dtype, squeeze=True)
        if return_indices:
            return data, idx
        else:
            return data


class OfflineEntry(object):
    """
    Simple object that contains a file pointer, and a selector callback. Don't have the *shape* attribute if not loaded
    once first
    """

    def __repr__(self):
        return "OfflineEntry(selector: %s, file: %s)" % (self.selector, self.file)

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = (item,)
        if len(item) == 0:
            return self
        else:
            return type(self)(self.file, selector=self.selector.__getitem__(*item), dtype=self._dtype, shape=self._pre_shape,
                        strides=self.strides)

    def __init__(self, file, selector=Selector(), dtype=None, shape=None, strides=None, **kwargs):
        """
        :param file: numpy file to load (.npy / .npz)
        :type file: str
        :param func: callback loading the file
        :type func: function
        :param dtype: optional cast of loaded data
        :type dtype: numpy.dtype
        :param target_length: specifies a target_length for the imported data, pad/cut in case
        :type target_length: int
        """

        self._pre_shape = shape
        self.metadata = kwargs
        if issubclass(type(file), OfflineEntry):
            self.file = file.file
            self.selector = selector or file.selector
            self.strides = strides or file.strides
            self._pre_shape = file.shape
            self._post_shape = self.selector.get_shape(file._pre_shape)
            self._dtype = file.dtype
            self.drop_time = file.drop_time
        else:
            self.file = file
            self.selector = selector
            self.strides = strides
            self._dtype = dtype

        if shape is not None:
            if self.selector is not None:
                self._post_shape = self.selector.get_shape(self._pre_shape)
            else:
                self._post_shape = self._pre_shape

    def __call__(self, file=None, selector=None, return_metadata=False, return_indices=False):
        """
        loads the file and extracts data
        :param file: optional file path
        :type file: str
        :return: array of data
        """
        file = file or self.file
        selector = selector or self.selector
        if return_indices:
            data, idx = selector(file, shape=self._pre_shape, dtype=self.dtype, strides=self.strides, return_indices=return_indices)
        else:
            data = selector(file, shape=self._pre_shape, dtype=self.dtype, strides=self.strides, return_indices=return_indices)

        if data.shape != self.shape:
            try:
                data = np.reshape(data, self.shape)
            except:
                raise ShapeError(data.shape, self.shape)

        if self._post_shape is None:
            self._post_shape = data.shape
        if self._dtype is None:
            data = data._dtype

        if return_metadata:
            metadata = self.metadata
            if return_indices:
                metadata['idx'] = idx
            return data, metadata
        else:
            return data

    @property
    def shape(self):
        shape = self._post_shape or self._pre_shape
        if shape is None:
            self()
            # raise NoShapeError('OfflineEntry has to be called once to retain shape')
        return self._post_shape

    @property
    def dtype(self):
        if self._dtype is None:
            raise TypeError('type of entry %s is not known yet' % self)
        return self._dtype

    def scatter(self, axis=0):
        if axis < 0:
            axis = len(self._pre_shape) + axis
        scatter_shape = self._pre_shape[axis]
        entries = [type(self)(self.file, self.selector.__getitem__(*((slice(None, None, None),)*axis + (i, ))), dtype=self._dtype,
                              shape=self._pre_shape, strides=self.strides) for i in range(scatter_shape)]
        return entries



class OfflineDataList(object):
    EntryClass = OfflineEntry

    def __repr__(self):
        string = '[\n'
        for e in self.entries:
            string += f"\t{str(e)},\n"
        string += ']'
        return string

    def __iter__(self):
        for entry in self.entries:
            yield entry()

    def __len__(self):
        return len(self.entries)

    @property
    def shape(self):
        if self._shape is None or self._update_shape:
            try:
                shapes = [e.shape for e in self.entries]
            except NoShapeError:
                self.check_entries(self.entries)
                shapes = [e.shape for e in self.entries]
            if is_same(shapes):
                self._shape = (len(self.entries), *shapes[0])
            self._update_shape = False
        return self._shape

    @property
    def ndim(self):
        if self.shape is not None:
            return len(self.shape)
        else:
            dims = set([e.shape for e in self.entries])
            if len(dims) > 1:
                raise Exception('cannot return ndim of OfflineDataList, found dimensions : %s'%dims)


    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = self[:].dtype
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = np.dtype(value)

    def __radd__(self, l):
        return l + self.entries

    def __setitem__(self, idx, elt):
        if issubclass(type(elt), OfflineEntry):
            if not issubclass(type(idx), tuple):
                self.entries[idx] = elt
                self._update_shape = True
            else:
                raise IndexError('cannot set elements in offline files')
        else:
            raise IndexError('class of OfflineDataList elements must be OfflineEntry')

    def __getitem__(self, args, return_metadata=None, return_indices=None):
        return_metadata = self.return_metadata if return_metadata is None else return_metadata
        return_indices = self.return_indices if return_indices is None else return_indices
        if isinstance(args, numbers.Integral):
            data = self.entries[args](return_metadata=self.return_metadata)
        elif isinstance(args, tuple):
            idx = args[0]
            if isinstance(args[0], numbers.Integral):
                data = self.entries[idx][args[1:]](return_metadata=return_metadata, return_indices=return_indices)
            if hasattr(args[0], "__iter__"):
                data = [self.entries[i][args[1:]](return_metadata=return_metadata, return_indices=return_indices) for i in idx]
            elif isinstance(args[0], slice):
                data = [self.entries[i][args[1:]](return_metadata=return_metadata, return_indices=return_indices) for i in range(len(self.entries))[idx]]
            else:
                raise IndexError("first index : %s invalid"%args[0])
        elif isinstance(args, slice):
            data = [entry(return_metadata=return_metadata, return_indices=return_indices) for entry in self.entries[args]]
        else:
            raise IndexError("Cannot parse OfflineDataList with index : %s "%args)

        if return_metadata:
            if isinstance(data, list):
                metadata = [d[1] for d in data]
                data = [d[0] for d in data]
            else:
                data, metadata = data

        # apply transforms
        if isinstance(data, list):
            data = [self.apply_transforms(d) for d in data]
        else:
            data = self.apply_transforms(data)

        data = batch_hash[self._batch_mode](data, **self.batch_args)
        if return_metadata:
            return data, metadata
        else:
            return data

    def apply_transforms(self, data):
        for t, a in self._transforms:
            data = t(data, **a)
        return data

    def take(self, ids):
        """
        return the entries object at given ids
        :param ids: entries ids to be picked
        :type ids: iterable
        :return: list(OfflineEntry)
        """
        if hasattr(ids, '__iter__'):
            target_entries = [self.entries[i] for i in ids]
            entry_list = type(self)(target_entries, transforms=self._transforms)
            return entry_list
        else:
            return self.entries[ids]

    def check_entries(self, entries):
        """
        check all the entries in the given list (an entry is considered wrong is it returns None once called)
        :param entries: list of entries to check
        :type entries: list(OfflineEntry)
        """
        invalid_entries = list(filter(lambda x: entries[x]() is None, range(len(entries))))
        for i in reversed(invalid_entries):
            del entries[i]

    def scatter(self, dim):
        new_entries = sum([x.scatter(0) for x in self.entries])
        return OfflineDataList(new_entries)

    def squeeze(self, dim):
        """
        add a squeeze transformation in the OfflineDataList
        :param dim: dimension to squeeze
        """
        if self._shape is None:
            raise Warning('Tried to squeeze %s, but shape is missing')
        if self._shape[dim] != 1:
            raise ValueError('cannot select an axis to squeeze out which has size not equal to one')
        if dim >= len(self._shape):
            raise np.AxisError('axis 4 is out of bounds for array of dimension 3' % (dim, len(self._shape)))
        self._transforms.append((np.squeeze, {'axis': dim}))
        new_shape = list(self._shape)
        del new_shape[dim]
        self._shape = tuple(new_shape)

    def expand_dims(self, dim):
        """
        add an unsqueeze transformation in the OfflineDataList
        :param dim: dimension to squeeze
        """
        if self._shape is None:
            print('Tried to squeeze %s, but shape is missing')
        self._transforms.append((np.expand_dims, {'axis': dim}))
        if self._shape is not None:
            if dim >= 0:
                self._shape = (*self._shape[:dim], 1, *self._shape[dim:])
            else:
                self._shape = (*self._shape[:dim + 1], 1, *self._shape[dim + 1:])

    @property
    def batch_mode(self):
        return self._batch_mode

    @batch_mode.setter
    def batch_mode(self, mode):
        assert mode in ['list', 'pad', 'crop']
        self._batch_mode = mode

    @property
    def ndim(self):
        if self._shape is None:
            return None
        else:
            return len(self._shape)


    def __init__(self, *args, check=False, dtype=None, transforms=None, selector=Selector(),
                 batch_mode="list", batch_args = {}, return_metadata=False, return_indices=False):
        """
        OfflineDataList is suupposed to be a ordered list of :class:`OfflineEntry`, that are callback objects reading files
        and pick data inside only once they are called. :class:`OfflineDataList` dynamically loads requested data when function
        :function:`__getitem__`is loaded, such that it can replace a casual np.array. Some functions of np.array are also
        overloaded such that squeeze / unsqueeze, based a sequence of data transforms that are applied once the data is
        loaded.
        :param args:
        :param selector_gen:
        :param check:
        :param dtype:
        :param stride:
        :param transforms:
        :param padded:
        """
        self._shape = None
        self._update_shape = True
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._transforms = []
        self.batch_mode = batch_mode
        self.batch_args = batch_args
        self.return_metadata = return_metadata
        self.return_indices = return_indices

        if len(args) == 1:
            if isinstance(args[0], dict):
                entries = []
                shapes = []
                for filename, meta in args[0].items():
                    entries.append(self.EntryClass(filename, selector=selector, **meta))
                    shapes.append(meta['shape'])
                self._shape = (len(shapes),) + tuple(get_full_shape(shapes))
                self.entries = entries
            elif isinstance(args[0], OfflineDataList):
                self.entries = list(args[0].entries)
                self._transforms = list(args[0]._transforms)
                self._shape = list(args[0]._shape)
                self._update_shape = False
                if dtype is None:
                    self._dtype = args[0]._dtype
            elif isinstance(args[0], list):
                entries = []
                for elt in args[0]:
                    if issubclass(type(elt), self.EntryClass):
                        entries.append(elt)
                    elif issubclass(type(elt), OfflineDataList):
                        entries.extend(elt.entries)
                self.entries = entries
                self._update_shape = True
            else:
                raise ValueError('expected OfflineDataList, dict or list, but got : %s' % type(args[0]))

        if check:
            self.check_entries(self.entries)

        if transforms is not None:
            self._transforms = transforms

        _ = self.shape

