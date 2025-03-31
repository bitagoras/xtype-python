# xtype - Python module

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/bitagoras/xtype-python)

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

## API Reference

### XTypeFile

The main class for reading and writing data in XType format.

#### Constructor

```python
XTypeFile(filename: str, mode: str = 'r')
```

- `filename`: Path to the file
- `mode`: File mode ('w' for write, 'r' for read)

#### Methods

- `write(data)`: Serialize and write a Python object to the file
- `read(byteorder='auto')`: Read and deserialize data from the file
- `read_debug(indent_size=2, max_indent_level=10, byteorder='auto', max_binary_bytes=15)`: Read and format data for debugging


## Project Links

- GitHub: [https://github.com/bitagoras/xtype](https://github.com/bitagoras/xtype)
