# xtype - Python library  <img src="doc/logo_xtype.png" width="50" align="right">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.0-green.svg)](https://github.com/bitagoras/xtype-python)

xtype is a Python library for serializing and deserializing data structures using the [xtype](https://github.com/bitagoras/xtype) binary format, optimized for efficient data exchange and storage.

## Features

- **Compact Binary Format**: Efficiently serialize data with minimal overhead
- **Type Preservation**: Maintains original data types during serialization and deserialization
- **Support for Complex Data Structures**:
  - Basic types: `int`, `float`, `str`, `bytes`, `bool`, `None`
  - Container types: `list`, `dict`
  - NumPy arrays: 1D, 2D, and higher-dimensional arrays with various data types
- **Cross-Platform Compatibility**: Consistent binary representation across different systems

## Quick Start

### Basic Usage

```python
import xtype
import numpy as np

# Sample data with various types
data = {
    "text": ["hello", "world"],
    "numeric_values": {
        "integer": 42,
        "float": 3.14159265359,
        "large_int": 9223372036854775807  # 2^63 - 1
    },
    "mixed_data_types": [True, False, None, [7, 7.7]],
    "binary_data": b"Binary data example",
    "array_data": np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32)
}

# Write data to file
with xtype.File("xtype-data.bin", 'w') as xf:
    xf.write(data)

# Read data from file
with xtype.File("xtype-data.bin", 'r') as xf:
    read_data = xf.read()
    print(read_data)
```

The data stored in `xtype-data.bin` has 208 Bytes.

### Debug Mode

xtype provides a debug mode to inspect the binary format:

```python
with xtype.File("xtype-data.bin", 'r') as xf:
    for chunk in xf.read_debug():
        print(chunk)
```

## Element Access Features

xtype supports efficient element access through indexing and slicing, allowing you to read specific elements from arrays and nested structures without loading the entire data set into memory.

### Accessing Dictionary Elements

```python
# Access dictionary values by key
with xtype.File("data.bin", 'r') as xf:
    # Access a top-level element
    value = xf["key"]()  # The () converts to Python object

    # Access nested dictionary elements
    nested_value = xf["parent"]["child"]()
```

### Accessing List and Array Elements

```python
with xtype.File("data.bin", 'r') as xf:
    # Access list items by index
    first_item = xf["my_list"][0]()  # () converts the object to Python

    # Access 2D array elements
    array_item = xf["my_array"][0,0]  # Note that no () is needed for array elements

    # Access 3D array elements
    value = xf["array_3d"][0,0,1]  # Second element in first row and column
```

### Using Slices for Lists and Arrays

```python
with xtype.File("data.bin", 'r') as xf:
    # List slices
    slice1 = xf["my_list"][1:4]  # Elements 1 through 3
    slice2 = xf["my_list"][::2]  # Every other element

    # Array slices
    # Get a slice of array with multiple dimensions
    subset = xf["my_3d_array"][0:2, 1:3, 2:4]

    # Slices with steps
    stepped_slice = xf["my_array"][0, 0, ::2]  # Every other element in 3rd dimension

    # Negative indices
    end_slice = xf["my_array"][0, 0, -2:]  # Last two elements
```

### Mixed Indexing with Arrays

```python
with xtype.File("data.bin", 'r') as xf:
    # Mixed integer and slice indexing
    row_slice = xf["my_3d_array"][0, 1, :]  # All elements in a specific row
```

## API Reference

### XTypeFile

The main class for reading and writing data in XType format.

#### Constructor

```python
xtype.File(filename: str, mode: str = 'r')
```

- `filename`: Path to the file
- `mode`: File mode ('w' for write, 'r' for read)

#### Methods

- `write(data)`: Serialize and write a Python object to the file
- `read()`: Read and deserialize data from the file
- `read_debug(indent_size=2, max_indent_level=10, max_binary_bytes=15)`: Read and format output for debugging
- `keys()`: Return a list of keys if the root object is a dictionary
- `__len__()`: Return the number of items in a list or dictionary, or the first dimension size of an array
- `__getitem__(key)`: Access an element within the file (supports integer indexing, slicing, and dictionary key access)
- `__setitem__(key, value)`: Assign a value to an element within the file (supports array assignment with integer indices, slices, and multi-dimensional indexing)


## Project Links

- GitHub: [https://github.com/bitagoras/xtype](https://github.com/bitagoras/xtype)
