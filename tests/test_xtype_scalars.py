"""
Unit tests for xtype library - NumPy scalar functionality tests

This test file focuses on NumPy scalar serialization and deserialization.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

# Add the lib directory to the path to import xtype
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/lib")
import xtype


@pytest.fixture
def temp_file():
    """Set up a temporary file for tests."""
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.close()
    yield temp
    # Clean up temporary files after tests
    if os.path.exists(temp.name):
        os.unlink(temp.name)

def test_numpy_scalars(temp_file):
    """Test serializing and deserializing NumPy scalar values."""
    test_data = {
        # Integer scalar types
        "int8_scalar": np.int8(-128),
        "int16_scalar": np.int16(-32768),
        "int32_scalar": np.int32(-2147483648),
        "int64_scalar": np.int64(-9223372036854775808),
        "uint8_scalar": np.uint8(255),
        "uint16_scalar": np.uint16(65535),
        "uint32_scalar": np.uint32(4294967295),
        "uint64_scalar": np.uint64(18446744073709551615),

        # Floating point scalar types
        "float16_scalar": np.float16(-1.0),
        "float32_scalar": np.float32(3.14159),
        "float64_scalar": np.float64(2.71828),

        # Boolean scalar type
        "bool_scalar": np.bool_(True),

        # Mixed data (scalar and array)
        "mixed_data": {
            "array": np.array([1, 2, 3], dtype=np.int32),
            "scalar": np.float64(3.14159)
        }
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    for key, value in test_data.items():
        if key != "mixed_data":
            assert read_data[key] == value
            # assert type(read_data[key]) == type(value)
        else:
            # Handle nested dictionary case
            np.testing.assert_array_equal(read_data[key]["array"], value["array"])
            assert read_data[key]["scalar"] == value["scalar"]
            # assert type(read_data[key]["scalar"]) == type(value["scalar"])

def test_scalar_conversion(temp_file):
    """Test that NumPy scalars are correctly converted in various contexts."""
    # Create data with NumPy scalars as dictionary keys and values
    test_data = {
        "dict_with_numpy_keys": {
            str(np.int32(1)): "value1",
            str(np.int64(2)): "value2"
        },
        "numpy_scalar_in_list": [np.int32(1), np.float64(2.5), np.bool_(True)]
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Verify that the keys were converted to strings (as required by xtype)
    assert read_data["dict_with_numpy_keys"]["1"] == "value1"
    assert read_data["dict_with_numpy_keys"]["2"] == "value2"

    # Verify that the list values match (may be Python native types on read)
    assert read_data["numpy_scalar_in_list"][0] == 1
    assert read_data["numpy_scalar_in_list"][1] == 2.5
    assert read_data["numpy_scalar_in_list"][2] == True
