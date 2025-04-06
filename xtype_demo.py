"""
XType Format Demo Script

This script demonstrates the usage of the XType binary format for serializing
Python data structures to files and reading raw data from xtype files.
"""

import numpy as np
import os

import xtype

print("XType Format Demo")
print("=================")

test_file = "test_xtype.bin"

# Sample data with various types
test_data = {
    "text": ["hello", "world"],
    "numeric_values": {
        "integer": 42,
        "float": 3.14159265359,
        "large_int": 9223372036854775807  # 2^63 - 1
    },
    "basic_data_types": [True, False, None, [7, 7.7]],
    "binary_data": b"Binary data example",
    "float_array": np.array([1.5, 2.5, 3.5], dtype=np.float64),
    "2d_array": np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32),
    # 2D string array example (2x3 array with fixed-length strings of 8 characters)
    "string_array_2d": np.array([["apple", "banana", "cherry"],
                                ["orange", "grape", "lemon"]], dtype="S8")
}

# Write data to file
print("\nWriting test data to file...")
with xtype.File(test_file, 'w') as xf:
    xf.write(test_data)

print(f"File size: {os.path.getsize(test_file)} bytes")

print("\nReading file in raw debug mode:")
with xtype.File(test_file, 'r') as xf:
    for chunk in xf.read_debug():
        print(chunk)

print("\nOriginal test data:")
print(test_data)

# Read data back using the new read method
print("\nReading data using read method:")
with xtype.File(test_file, 'r') as xf:
    read_data = xf.read()
print(read_data)

print("\nDemo completed successfully!")

# Demonstrate the new objPointer functionality
print("\n\nObjPointer Feature Demo")
print("======================")

with xtype.File(test_file, 'r') as xf:
    print("\n0. Show dictionary keys:")
    print(xf.keys())

    print("\n1. Access dictionary values by key:")
    # Get a specific dictionary key using __getitem__
    text_value = xf["text"]()  # Call to convert to Python object
    print(f"Text value: {text_value}")

    print("\n2. Access nested dictionary values:")
    print("get whole subitem:")
    integer_value = xf["numeric_values"]()
    print(integer_value)
    # Navigate through nested structures
    print("get sub-subitem")
    integer_value = xf["numeric_values"]["integer"]()
    print(f"Integer value: {integer_value}")
    float_value = xf["numeric_values"]["float"]()
    print(f"Float value: {float_value}")

    print("\n3. Access list elements by index:")
    # Get specific list elements
    hello_value = xf["text"][0]()
    print(f"First text item: {hello_value}")
    world_value = xf["text"][1]()
    print(f"Second text item: {world_value}")

    # print("\n4. Access elements in arrays:")
    # # Access array elements
    # first_float = xf["float_array"][0]()
    # print(f"First float in array: {first_float}")

    # # Access 2D array elements (using chained __getitem__ calls)
    # nested_value = xf["2d_array"][0][0][1]()
    # print(f"Value at [0][0][1] in 2D array: {nested_value}")

    # print("\n5. Navigate complex nested structures:")
    # # Chain multiple navigation steps
    # nested_list_item = xf["basic_data_types"][3][1]()
    # print(f"Nested list item at basic_data_types[3][1]: {nested_list_item}")

    # print("\n6. Demonstrate performance benefit for large structures:")
    # # Only convert the part of the structure we need
    # print("Converting only specific parts of the data structure:")
    # specific_array = xf["2d_array"]()
    # print(f"2D array: {specific_array}")

    # # For comparison (already did this earlier with full read)
    # print("\nObjPointer feature demo completed successfully!")
