"""
XType Format Demo Script

This script demonstrates the usage of the XType binary format for serializing
Python data structures to files and reading raw data from xtype files.
Including examples of the advanced array indexing functionality.
"""

import numpy as np
import os

import xtype

print("XType Format Demo")
print("=================")

#---------------------------------
# 1. Basic Data Serialization Demo
#---------------------------------
print("\n1. Basic Data Serialization Demo")
print("------------------------------")

test_file = "xtype_test_data.bin"

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
    "array_3d": np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32),
    # 2D string array example (2x3 array with fixed-length strings of 8 characters)
    "string_array_2d": np.array([["apple", "banana", "cherry"],
                                ["orange", "grape", "lemon"]], dtype="S8")
}

# Write data to file
print("\n1.1 Writing test data to file...")
with xtype.File(test_file, 'w') as xf:
    xf.write(test_data)

print(f"File size: {os.path.getsize(test_file)} bytes")

print("\n1.2 Reading file in raw debug mode:")
with xtype.File(test_file, 'r') as xf:
    for chunk in xf.read_debug():
        print(chunk)

print("\n1.3 Original test data:")
print(test_data)

# Read data back using the new read method
print("\n1.4 Reading data using read method:")
with xtype.File(test_file, 'r') as xf:
    read_data = xf.read()
print(read_data)

#--------------------------------
# 2. ObjPointer Navigation Demo
#--------------------------------
print("\n\n2. ObjPointer Navigation Demo")
print("------------------------------")

with xtype.File(test_file, 'r') as xf:
    print("\n2.1 Show dictionary length and keys:")
    print(len(xf), xf.keys())

    print("\n2.2 Access dictionary values by key:")
    # Get a specific dictionary key using __getitem__
    text_value = xf["text"]()  # Call to convert to Python object
    print(f"Text value: {text_value}")

    print("\n2.3 Access nested dictionary values:")
    print("Get whole subitem:")
    dict_value = xf["numeric_values"]()
    print(dict_value)

    # Navigate through nested structures
    print("Get sub-subitem:")
    integer_value = xf["numeric_values"]["integer"]()
    print(f"Integer value: {integer_value}")
    float_value = xf["numeric_values"]["float"]()
    print(f"Float value: {float_value}")

    print(f"\n2.4 Access list elements (len = {len(xf['text'])}) by index:")
    # Get specific list elements
    hello_value = xf["text"][0]()
    print(f"First text item: {hello_value}")
    world_value = xf["text"][1]()
    print(f"Second text item: {world_value}")

    print("\n2.5 Basic array element access:")
    # Access array elements
    first_float = xf["float_array"][0]
    print(f"First float in array: {first_float}")

    # Access 2D sub array (using chained __getitem__ calls)
    nested_value = xf["array_3d"][0]
    print(f"Value at [0] in 3D array: {nested_value}")

    # Access 1D array elements (using chained __getitem__ calls)
    nested_value = xf["array_3d"][0,0]
    print(f"Value at [0,0] in 3D array: {nested_value}")

    # Access single element (using chained __getitem__ calls)
    nested_value = xf["array_3d"][0,0,1]
    print(f"Value at [0,0,1] in 3D array: {nested_value}")

    print("\n2.6 List slicing example:")
    # Demonstrate list slicing on the text list
    list_slice = xf["text"][0:2]  # Get all elements
    print(f"Full list slice xf[\"text\"][0:2]: {list_slice}")

    print("\n2.7 Lenght of list:")
    # Len of array
    print(f'len(xf["basic_data_types"]): {len(xf["basic_data_types"])}')

    print("\n2.8 Navigate complex nested structures:")
    # Chain multiple navigation steps
    nested_list_item = xf["basic_data_types"][3][1]()
    print(f"Nested list item at basic_data_types[3][1]: {nested_list_item}")
    # Demonstrate step parameter in list slicing
    step_slice = xf["basic_data_types"][0:4:2]  # Get elements with step -2
    print(f"Slice with step xf[\"basic_data_types\"][0:4:2]: {step_slice}")
    step_slice = xf["basic_data_types"][4:0:-2]  # Get elements with step -2
    print(f"Slice with step xf[\"basic_data_types\"][4:0:-2]: {step_slice}")


    print("\n2.9 Demonstrate performance benefit for large structures:")
    # Only convert specific parts of the structure without loading everything
    print("Converting only specific parts of the data structure:")
    specific_array = xf["array_3d"]()
    print(f"3D array: {specific_array}")

#---------------------------------
# 3. Advanced Array Indexing Demo
#---------------------------------
print("\n\n3. Advanced Array Indexing Demo")
print("----------------------------")

# Create a new file with a larger multi-dimensional array for better demonstration
array_test_file = "xtype_test_array.bin"

# Create a 4D array (3x4x5x6) with sequential integers
array_data = {
    "array_4d": np.arange(360).reshape(3, 4, 5, 6)
}

print("\n3.1 Creating test file with 4D array...")
with xtype.File(array_test_file, 'w') as xf:
    xf.write(array_data)

with xtype.File(array_test_file, 'r') as xf:
    print("\n3.2 Access with partial indexing (fewer indices than dimensions):")
    # Example: A[1,2] returns array of shape (5,6)
    subset1 = xf["array_4d"][1, 2]
    print(f"xf[\"array_4d\"][1, 2]")
    print(f"Shape: {subset1.shape}")
    print(f"First few values: {subset1[0, 0:3]}...")

    print("\n3.3 Access with slice indexing:")
    # Example: A[1:,2,3] returns array of shape (2,6)
    subset2 = xf["array_4d"][1:, 2, 3]
    print(f"xf[\"array_4d\"][1:, 2, 3]")
    print(f"Shape: {subset2.shape}")
    print(f"Values: \n{subset2}")

    print("\n3.4 Access full array with [:]:")
    # Example: A[:] returns the full array
    full_array = xf["array_4d"][:]
    print(f"xf[\"array_4d\"][:]")
    print(f"Shape: {full_array.shape}")
    print(f"First five elements: {full_array[0, 0, 0, :5]}")

    print("\n3.5 Access with step in slice:")
    # Example: A[0,0,0,::2] returns array of shape (3)
    subset3 = xf["array_4d"][0, 0, 0, ::2]
    print(f"xf[\"array_4d\"][0, 0, 0, ::2]")
    print(f"Shape: {subset3.shape}")
    print(f"Values: {subset3}")

    print("\n3.6 Access with list indexing:")
    # Example: A[0, [0,2], 1] returns array with selected indices from dimension 1
    subset4 = xf["array_4d"][0, [0, 2], 1]
    print(f"xf[\"array_4d\"][0, [0, 2], 1]")
    print(f"Shape: {subset4.shape}")
    print(f"Values: \n{subset4}")

    print("\n3.7 Complex slicing example:")
    # Example: A[0:2, 1:3, 2:4, 1:5:2] returns a complex subset
    subset5 = xf["array_4d"][0:2, 1:3, 2:4, 1:5:2]
    print(f"Shape of complex slice: ")
    print(f"Shape: {subset5.shape}")
    print(f"Values first part): \n{subset5[0, 0]}")

    print("\n3.8 Combination of integer and slice indexing:")
    # Example: Mixed indexing approaches
    subset6 = xf["array_4d"][0, 1:3, :, 2]
    print(f"Shape: {subset6.shape}")
    print(f"Values: \n{subset6[0]}")

    print("\n3.9 Negative indices in slices:")
    # Example: Using negative indices
    subset7 = xf["array_4d"][0, 0, -2:, -3:-1]
    print(f"Shape: {subset7.shape}")
    print(f"Values: \n{subset7}")

print("\nDemo completed successfully!")

# Output:

# XType Format Demo
# =================

# 1. Basic Data Serialization Demo
# ------------------------------

# 1.1 Writing test data to file...
# File size: 315 bytes

# 1.2 Reading file in raw debug mode:
# *j: 04 d2
# {
#   4s: "text"
#   [
#     5s: "hello"
#     5s: 77 6f 72 6c 64
#   ]
#   M(14)s: 6e 75 6d 65 72 69 63 5f 76 61 6c 75 65 73
#   {
#     7s: 69 6e 74 65 67 65 72
#     I: 2a
#     5s: 66 6c 6f 61 74
#     d: 40 09 21 fb 54 44 2e ea
#     9s: 6c 61 72 67 65 5f 69 6e 74
#     L: 7f ff ff ff ff ff ff ff
#   }
#   M(16)s: 62 61 73 69 63 5f 64 61 74 61 5f 74 79 70 65 ... (16 bytes total)
#   [
#     TFn
#     [
#       I: 07
#       d: 40 1e cc cc cc cc cc cd
#     ]
#   ]
#   M(11)s: 62 69 6e 61 72 79 5f 64 61 74 61
#   M(19)x: 42 69 6e 61 72 79 20 64 61 74 61 20 65 78 61 ... (19 bytes total)
#   M(11)s: 66 6c 6f 61 74 5f 61 72 72 61 79
#   3d: 3f f8 00 00 00 00 00 00 40 04 00 00 00 00 00 ... (24 bytes total)
#   8s: 61 72 72 61 79 5f 33 64
#   123k: 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 ... (24 bytes total)
#   M(15)s: 73 74 72 69 6e 67 5f 61 72 72 61 79 5f 32 64
#   238s: 61 70 70 6c 65 00 00 00 62 61 6e 61 6e 61 00 ... (48 bytes total)
# }

# 1.3 Original test data:
# {'text': ['hello', 'world'], 'numeric_values': {'integer': 42, 'float': 3.14159265359, 'large_int': 9223372036854775807}, 'basic_data_types': [True, False, None, [7, 7.7]], 'binary_data': b'Binary data example', 'float_array': array([1.5, 2.5, 3.5]), 'array_3d': array([[[1, 2, 3],
#         [4, 5, 6]]]), 'string_array_2d': array([[b'apple', b'banana', b'cherry'],
#        [b'orange', b'grape', b'lemon']], dtype='|S8')}

# 1.4 Reading data using read method:
# {'text': ['hello', 'world'], 'numeric_values': {'integer': 42, 'float': 3.14159265359, 'large_int': 9223372036854775807}, 'basic_data_types': [True, False, None, [7, 7.7]], 'binary_data': b'Binary data example', 'float_array': array([1.5, 2.5, 3.5]), 'array_3d': array([[[1, 2, 3],
#         [4, 5, 6]]]), 'string_array_2d': array([[b'apple', b'banana', b'cherry'],
#        [b'orange', b'grape', b'lemon']], dtype='|S8')}


# 2. ObjPointer Navigation Demo
# ------------------------------

# 2.1 Show dictionary keys:
# ['text', 'numeric_values', 'basic_data_types', 'binary_data', 'float_array', 'array_3d', 'string_array_2d']

# 2.2 Access dictionary values by key:
# Text value: ['hello', 'world']

# 2.3 Access nested dictionary values:
# Get whole subitem:
# {'integer': 42, 'float': 3.14159265359, 'large_int': 9223372036854775807}
# Get sub-subitem:
# Integer value: 42
# Float value: 3.14159265359

# 2.4 Access list elements by index:
# First text item: hello
# Second text item: world

# 2.5 Basic array element access:
# First float in array: 1.5
# Value at [0] in 3D array: [[1 2 3]
#  [4 5 6]]
# Value at [0,0] in 3D array: [1 2 3]
# Value at [0,0,1] in 3D array: 2

# 2.5.1 List slicing example:
# Full list slice xf["text"][0:2]: ['hello', 'world']
# Slice with step xf["basic_data_types"][0:4:2]: [True, None]
# Slice with step xf["basic_data_types"][4:0:-2]: [None, True]

# 2.6 Navigate complex nested structures:
# Nested list item at basic_data_types[3][1]: 7.7

# 2.7 Demonstrate performance benefit for large structures:
# Converting only specific parts of the data structure:
# 3D array: [[[1 2 3]
#   [4 5 6]]]


# 3. Advanced Array Indexing Demo
# ----------------------------

# 3.1 Creating test file with 4D array...

# 3.2 Access with partial indexing (fewer indices than dimensions):
# xf["array_4d"][1, 2]
# Shape: (5, 6)
# First few values: [180 181 182]...

# 3.3 Access with slice indexing:
# xf["array_4d"][1:, 2, 3]
# Shape: (2, 6)
# Values:
# [[120 121 122 123 124 125]
#  [126 127 128 129 130 131]]

# 3.4 Access full array with [:]:
# xf["array_4d"][:]
# Shape: (3, 4, 5, 6)
# First five elements: [0 1 2 3 4]

# 3.5 Access with step in slice:
# xf["array_4d"][0, 0, 0, ::2]
# Shape: (3,)
# Values: [0 2 4]

# 3.6 Access with list indexing:
# xf["array_4d"][0, [0, 2], 1]
# Shape: (2, 6)
# Values:
# [[ 0  1  2  3  4  5]
#  [60 61 62 63 64 65]]

# Shape of complex slice:
# Shape: (2, 2, 2, 2)
# Values first part):
# [[43 45]
#  [49 51]]

# 3.8 Combination of integer and slice indexing:
# Shape: (2, 5)
# Values:
# [30 31 32 33 34]

# 3.9 Negative indices in slices:
# Shape: (2, 2)
# Values:
# [[21 22]
#  [23 24]]

# Demo completed successfully!