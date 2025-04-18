"""
Unit tests for xtype library - Advanced features and edge cases

This test file covers more advanced features like byteorder handling,
large file operations, and other edge cases.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import struct

# Add the lib directory to the path to import xtype
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lib")
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

def test_byteorder_big(temp_file):
    """Test explicit big-endian byte order."""
    test_data = {
        "int32": np.array([123456789], dtype=np.int32)[0],
        "float64": 3.14159265359,
        "array": np.array([1.0, 2.0, 3.0], dtype=np.float64)
    }

    # Write data with big-endian byte order
    with xtype.File(temp_file.name, 'w', byteorder='big') as xf:
        xf.write(test_data)

    # Read data back with auto byte order detection
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    assert read_data["int32"] == test_data["int32"]
    assert read_data["float64"] == test_data["float64"]
    np.testing.assert_array_equal(read_data["array"], test_data["array"])

def test_byteorder_little(temp_file):
    """Test explicit little-endian byte order."""
    test_data = {
        "int32": np.array([123456789], dtype=np.int32)[0],
        "float64": 3.14159265359,
        "array": np.array([1.0, 2.0, 3.0], dtype=np.float64)
    }

    # Write data with little-endian byte order
    with xtype.File(temp_file.name, 'w', byteorder='little') as xf:
        xf.write(test_data)

    # Read data back with auto byte order detection
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    assert read_data["int32"] == test_data["int32"]
    assert read_data["float64"] == test_data["float64"]
    np.testing.assert_array_equal(read_data["array"], test_data["array"])

def test_large_integers(temp_file):
    """Test serializing and deserializing very large integers."""
    test_data = {
        "small_int": 42,
        "medium_int": 123456789,
        "large_int": 9223372036854775807,  # 2^63 - 1 (max int64)
        "very_large_int": 18446744073709551615  # 2^64 - 1 (max uint64)
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    assert read_data == test_data

def test_empty_file(temp_file):
    """Test handling of empty files."""
    # Create an empty file
    with open(temp_file.name, 'wb') as f:
        pass

    # Try to read from empty file
    with xtype.File(temp_file.name, 'r') as xf:
        assert xf is None  # Empty file should return None

def test_nested_structure_depth(temp_file):
    """Test deeply nested data structures."""
    # Create a deeply nested dictionary
    test_data = {}
    current = test_data
    for i in range(10):  # 10 levels of nesting
        current["level"] = {}
        current = current["level"]
    current["value"] = 42

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Navigate to the deepest level and check value
    current_orig = test_data
    current_read = read_data
    for i in range(10):
        assert "level" in current_read
        current_orig = current_orig["level"]
        current_read = current_read["level"]
    assert current_read["value"] == 42

def test_large_array(temp_file):
    """Test serializing and deserializing a large array."""
    # Create a large array (1 million elements)
    large_array = np.arange(1_000_000, dtype=np.int32)
    test_data = {"large_array": large_array}

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data - only check a few elements for performance
    np.testing.assert_array_equal(read_data["large_array"][:10], test_data["large_array"][:10])
    np.testing.assert_array_equal(read_data["large_array"][-10:], test_data["large_array"][-10:])
    assert read_data["large_array"].shape == test_data["large_array"].shape
    assert read_data["large_array"].dtype == test_data["large_array"].dtype

    # Test accessing specific elements
    with xtype.File(temp_file.name, 'r') as xf:
        # Access beginning elements
        assert xf["large_array"][0] == 0
        assert xf["large_array"][1] == 1
        
        # Access middle elements
        assert xf["large_array"][500_000] == 500_000
        
        # Access end elements
        assert xf["large_array"][-1] == 999_999
        assert xf["large_array"][-2] == 999_998

def test_file_operations(temp_file):
    """Test file operations and context management."""
    test_data = {"sample": "data"}

    # Test context manager
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Test manual open/close
    xf = xtype.File(temp_file.name, 'r')
    xf.open()
    assert xf.read() == test_data
    xf.close()

    # Test file is closed properly
    assert xf.file.closed

    # Test reusing the same file object
    xf = xtype.File(temp_file.name, 'r')
    xf.open()
    result = xf.read()
    xf.close()
    assert result == test_data

def test_mixed_complex_structures(temp_file):
    """Test complex mixed data structures with arrays of different shapes and types."""
    # Create complex nested structure
    test_data = {
        "scalars": {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "bool": True,
            "none": None
        },
        "arrays": {
            "1d_int": np.arange(10, dtype=np.int32),
            "2d_float": np.ones((3, 4), dtype=np.float64),
            "3d_bool": np.zeros((2, 3, 2), dtype=bool)
        },
        "mixed_list": [
            1,
            "two",
            np.array([3.0, 4.0], dtype=np.float32),
            {"nested": True}
        ]
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    # Need to check each part separately due to NumPy arrays
    # Scalars
    assert read_data["scalars"] == test_data["scalars"]

    # Arrays
    for key, value in test_data["arrays"].items():
        np.testing.assert_array_equal(read_data["arrays"][key], value)

    # Mixed list - special handling for NumPy array
    assert read_data["mixed_list"][0] == test_data["mixed_list"][0]
    assert read_data["mixed_list"][1] == test_data["mixed_list"][1]
    np.testing.assert_array_equal(
        read_data["mixed_list"][2],
        test_data["mixed_list"][2]
    )
    assert read_data["mixed_list"][3] == test_data["mixed_list"][3]

def test_keys_method(temp_file):
    """Test the keys() method for dictionary access."""
    test_data = {
        "a": 1,
        "b": 2,
        "c": {
            "d": 3,
            "e": 4
        }
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read back and test keys method
    with xtype.File(temp_file.name, 'r') as xf:
        # Test root keys
        assert set(xf.keys()) == {"a", "b", "c"}

        # Test nested keys
        assert set(xf["c"].keys()) == {"d", "e"}

def test_len_method(temp_file):
    """Test the __len__ method for dictionaries and lists."""
    test_data = {
        "dict": {"a": 1, "b": 2, "c": 3},
        "list": [1, 2, 3, 4, 5],
        "nested": {
            "inner_dict": {"x": 1, "y": 2},
            "inner_list": [9, 8, 7]
        }
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read back and test len method
    with xtype.File(temp_file.name, 'r') as xf:
        # Test dictionary lengths
        assert len(xf["dict"]) == 3
        assert len(xf["nested"]["inner_dict"]) == 2

        # Test list lengths
        assert len(xf["list"]) == 5
        assert len(xf["nested"]["inner_list"]) == 3

        # Test root length (should be 3 keys)
        assert len(xf) == 3

        # Test len() raises TypeError for non-container types
        with pytest.raises(TypeError):
            len(xf["dict"]["a"])
