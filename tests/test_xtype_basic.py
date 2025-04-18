"""
Unit tests for xtype library - Basic functionality tests

This test file covers basic serialization/deserialization of primitive types,
lists, dictionaries, and basic error handling.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

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

def test_primitive_types(temp_file):
    """Test serializing and deserializing primitive types."""
    test_data = {
        "integer": 42,
        "negative_int": -17,
        "large_int": 9223372036854775807,  # 2^63 - 1
        "float": 3.14159265359,
        "boolean_true": True,
        "boolean_false": False,
        "none_value": None,
        "string": "Hello, world!",
        "bytes": b"Binary data"
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    assert read_data == test_data

    # Test individual element access
    with xtype.File(temp_file.name, 'r') as xf:
        assert xf["integer"] == 42
        assert xf["negative_int"] == -17
        assert xf["large_int"] == 9223372036854775807
        assert xf["float"] == 3.14159265359
        assert xf["boolean_true"] is True
        assert xf["boolean_false"] is False
        assert xf["none_value"] is None
        assert xf["string"] == "Hello, world!"
        assert xf["bytes"] == b"Binary data"

def test_container_types(temp_file):
    """Test serializing and deserializing container types (lists and dictionaries)."""
    test_data = {
        "empty_list": [],
        "list_of_ints": [1, 2, 3, 4, 5],
        "list_of_mixed": [1, "two", 3.0, True, None],
        "nested_list": [1, [2, 3], [4, [5, 6]]],
        "empty_dict": {},
        "simple_dict": {"a": 1, "b": 2, "c": 3},
        "nested_dict": {"a": {"b": {"c": 42}}}
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    assert read_data == test_data

    # Test list operations
    with xtype.File(temp_file.name, 'r') as xf:
        assert len(xf["list_of_ints"]) == 5
        assert xf["list_of_ints"][0] == 1
        assert xf["list_of_ints"][4] == 5
        assert xf["nested_list"][1][1] == 3
        assert xf["nested_list"][2][1][1] == 6

    # Test dict operations
    with xtype.File(temp_file.name, 'r') as xf:
        assert len(xf["simple_dict"]) == 3
        assert xf["simple_dict"]["a"] == 1
        assert xf["simple_dict"]["c"] == 3
        assert xf["nested_dict"]["a"]["b"]["c"] == 42

def test_slicing(temp_file):
    """Test list slicing operations."""
    test_data = {
        "list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Test slicing operations
    with xtype.File(temp_file.name, 'r') as xf:
        # Basic slices
        assert xf["list"][2:5] == [2, 3, 4]
        assert xf["list"][:3] == [0, 1, 2]
        assert xf["list"][7:] == [7, 8, 9]
        assert xf["list"][:] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Slices with steps
        assert xf["list"][::2] == [0, 2, 4, 6, 8]
        assert xf["list"][1::2] == [1, 3, 5, 7, 9]
        assert xf["list"][1:8:3] == [1, 4, 7]

def test_error_cases(temp_file):
    """Test error handling."""
    # Create a simple file with dictionary data
    test_data = {"key1": "value1", "key2": [1, 2, 3]}

    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Test different error cases
    with xtype.File(temp_file.name, 'r') as xf:
        # Key not found
        with pytest.raises(KeyError):
            xf["non_existent_key"]()

        # Index out of range
        with pytest.raises(IndexError):
            xf["key2"][5]()

        # Type error (trying to index a non-indexable object)
        with pytest.raises(TypeError):
            xf["key1"][0]()

def test_file_modes(temp_file):
    """Test file opening in different modes."""
    # Test write mode
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write({"test": 123})

    # Test read mode
    with xtype.File(temp_file.name, 'r') as xf:
        assert xf.read() == {"test": 123}

    # Test invalid mode
    with pytest.raises(ValueError):
        with xtype.File(temp_file.name, 'x') as xf:
            pass
