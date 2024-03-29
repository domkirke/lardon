# lardon
Lardon is a front-end for dynamic data import of large files, using the `numpy.memmap` interface 
to easily index large memory arrays without the entire import of the corresponding file.
## Setup
To install `lardon`, run you Terminal and 
```
git clone https://github.com/domkirke/lardon.git
cd lardon
python3 setup.py install 
```

## Quick start
To parse a dataset with `lardon`, you can use to different frontends:

```python
import lardon

data_path = "${path_to_dataset}"
parsed_path = "${path_to_parsed_dataset}"

# online parsing
offline_list = lardon.compile(data_path, parsed_path, valid_exts=['.npy'], callback=lardon.default_callback)

# environment parsing (data_path can be None, that you can generate data while exporting it. if you do so,
# the lardon parser will not have an iteration method)
with lardon.LardonParser(data_path, parsed_path, valid_exts=['.npy']) as parser:
    for f in parser:
        data = np.load(f)
        parser.register(f, {}, f)

# you can then load the offline list with the parse_folder method
offline_list = lardon.parse_folder(parsed_path, drop_metadata=True)
data, metadata = offline_list[0] # lardon is asynchronous, s.t. data will be loaded when accessed
```



## Description

Lardon first parses the target data to create a `memmap`-compatible
version of the dataset, referring to data elements using aysnchonous callback units called `OfflineEntry`, 
whose corresponding data is pointed using a `Selector`. Entries are compiled in an `OfflineDataList` object,
that can be indexed as an usual list. 

###### Warning
Please consider that, due to the `numpy.memmap` back-end, import time goes exponentially with the depth of the
sliced dimension. Hence, dynamic import of deep dimensions of an array. For exemple, if the array is 10x10x10x10 `x[:, :, :, 0]`
will result to 1000 `memmap` calls, while `x[:, 0:20]` will result to only 10 calls. 

### Data parsing
Data parsing can be easily done using the ``compile`` function :
```python
import numpy as np
from lardon import checkdir, compile

# Create dumb dataset
path = "dumb_dataset"
checkdir(path); checkdir(path+"/data")
n_items = 10
for n in range(n_items):
    random_shape = (10, np.random.randint(2, 5), 23, np.random.randint(700, 900))
    x = np.reshape(np.arange(np.prod(random_shape)), random_shape)
    with open(f"{path}/data/dumb_{n}.npy", 'wb') as f:
        np.save(f, x)

# Define callback for data import
def dumb_callback(filename):
    with open(filename, 'rb') as f:
        # callback must return data as first argument, and metadata as second
        return np.load(f), {}


# the compile function exports data as np.memmap and the corresponding OfflineDataList
offline_list = compile(path+'/data', path+"/parsed", valid_exts=['.npy'], callback=dumb_callback)
```  

### Data loading
A parsed dataset (a dataset will be considered as parsed if a file `parsing.ldn` is found
in the root) can be imported using a simple call to `parse_folder` : 

````python
from lardon import parse_folder

offline_list = parse_folder("dumb_dataset/parsed")
# an additional dtype argument may be given to convert (if needed) the dtype of the loaded data
print(offline_list[0].shape, offline_list[0, 5, 0:2, 1:2, 0].shape)
````

A given import axis can be chosen randomly by indexing with a `randslice(length=1)` index,
allowing to retrieve random parts of the imported sequence at each call: 
````python
from lardon import parse_folder, randslice

offline_list = parse_folder("dumb_dataset/parsed")
for i in range(5):
    # randslice object is initialized with a length argument (default is 1) and optional start and
    #    stop keywords, constraining the range used for random indexing
    print("iter %d : "%i, offline_list[0].shape, offline_list[0, 5, randslice(10, start=10, stop=50), 1:2, 0])
````
As ``numpy.memmap`` is used in the back-end, full import of the file is not needed anymore. However, please note that
the number of `memmap` calls grows exponentially with the depth of the dimension, such that is much more efficient
to index (randomly or not) the first dimensions of the imported array.


### Batched data
```lardon``` compiles all the files in a single data list, such that batch loading of variable sequences
may be not straightforward. The behavior of data batching may be set using the `batch_mode` keyword :

```python
from lardon import parse_folder

path = "/Users/domkirke/Datasets/lardon_dataset"
# the pad mode allows additional arguments for the np.pad function
offline_list = parse_folder(path+'/parsed', batch_mode="pad",
                            batch_args={'mode':'constant', 'constant_values':2})
batched_import = offline_list[:]
print(batched_import.shape)

offline_list = parse_folder(path+'/parsed', batch_mode="pad")
batched_import = offline_list[:]
print(batched_import.shape)
``` 

### Basic re-shaping
Basic reshaping of incoming data can also be performed using the `expand_dims` or `squeeze` methods.
Please note that numpy callbacks are called before data batching, such that the axis keyword refer to 
the axis of individual batches. 

```python
from lardon import parse_folder

path = "/Users/domkirke/Datasets/lardon_dataset"
# the pad mode allows additional arguments for the np.pad function
offline_list = parse_folder(path+'/parsed', batch_mode="pad",
                            batch_args={'mode':'constant', 'constant_values':2})
offline_list.expand_dims(2)
batched_import = offline_list[0]
print(batched_import.shape)
```






