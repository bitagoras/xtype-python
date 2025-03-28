"""
XType Format Demo Script

This script demonstrates the usage of the XType binary format for serializing
Python data structures to files and reading raw data from xtype files.
"""

import numpy as np
import os

from xtype import XTypeFile

print("XType Format Demo")
print("=================")

test_file = "test_xtype.bin"

# Sample data with various types
test_data = {
    "version": 1.0,
    "text": ["hello", "world"],
    "numeric_values": {
        "integer": 42,
        "float": 3.14159265359,
        "large_int": 9223372036854775807  # 2^63 - 1
    },
    "basic_data_types": [True, False, None, [7, 7.7]],
    "binary_data": b"Binary data example",
    "array_data": np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32),
    "mixed_array": np.array([1.5, 2.5, 3.5], dtype=np.float64),
    # 2D string array example (2x3 array with fixed-length strings of 8 characters)
    "string_array_2d": np.array([["apple", "banana", "cherry"],
                                ["orange", "grape", "lemon"]], dtype="S8")
}

# Write data to file
print("\nWriting test data to file...")
with XTypeFile(test_file, 'w') as xf:
    xf.write(test_data)

print(f"File size: {os.path.getsize(test_file)} bytes")

print("\nReading file in raw debug mode:")
with XTypeFile(test_file, 'r') as xfile:
    for chunk in xfile.read_debug():
        print(chunk)

print("\nOriginal test data:")
print(test_data)

# Read data back using the new read method
print("\nReading data using read method:")
with XTypeFile(test_file, 'r') as xfile:
    read_data = xfile.read()
print(read_data)

print("\nDemo completed successfully!")
