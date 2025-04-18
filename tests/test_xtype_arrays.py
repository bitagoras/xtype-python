"""
Unit tests for xtype library - NumPy array functionality tests

This test file focuses on NumPy array serialization, deserialization,
and the array indexing functionality including multi-dimensional arrays.
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

def create_test_file_with_arrays(temp_file):
    """Helper method to create a test file with various arrays."""
    test_data = {
        "array_1d": np.arange(10, dtype=np.int32),
        "array_2d": np.reshape(np.arange(12, dtype=np.float32), (3, 4)),
        "array_3d": np.reshape(np.arange(24, dtype=np.int16), (2, 3, 4)),
        "array_4d": np.reshape(np.arange(120, dtype=np.int8), (2, 3, 4, 5))
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    return test_data

def test_1d_arrays(temp_file):
    """Test serializing and deserializing 1D NumPy arrays of different types."""
    test_data = {
        "int8_array": np.array([-128, 0, 127], dtype=np.int8),
        "int16_array": np.array([-32768, 0, 32767], dtype=np.int16),
        "int32_array": np.array([-2147483648, 0, 2147483647], dtype=np.int32),
        "int64_array": np.array([-9223372036854775808, 0, 9223372036854775807], dtype=np.int64),
        "uint8_array": np.array([0, 128, 255], dtype=np.uint8),
        "uint16_array": np.array([0, 32768, 65535], dtype=np.uint16),
        "uint32_array": np.array([0, 2147483648, 4294967295], dtype=np.uint32),
        "uint64_array": np.array([0, 9223372036854775808, 18446744073709551615], dtype=np.uint64),
        "float32_array": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "float64_array": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        "bool_array": np.array([True, False, True], dtype=bool)
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    for key, value in test_data.items():
        np.testing.assert_array_equal(read_data[key], value)
        assert read_data[key].dtype == value.dtype

def test_2d_arrays(temp_file):
    """Test serializing and deserializing 2D NumPy arrays."""
    test_data = {
        "int32_2d": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "float64_2d": np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float64),
        "bool_2d": np.array([[True, False], [False, True]], dtype=bool)
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    for key, value in test_data.items():
        np.testing.assert_array_equal(read_data[key], value)
        assert read_data[key].dtype == value.dtype
        assert read_data[key].shape == value.shape

def test_multidimensional_arrays(temp_file):
    """Test serializing and deserializing multi-dimensional NumPy arrays."""
    test_data = {
        "3d_array": np.arange(60).reshape(3, 4, 5),
        "4d_array": np.arange(120).reshape(2, 3, 4, 5)
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    for key, value in test_data.items():
        np.testing.assert_array_equal(read_data[key], value)
        assert read_data[key].dtype == value.dtype
        assert read_data[key].shape == value.shape

def test_string_arrays(temp_file):
    """Test serializing and deserializing string arrays."""
    test_data = {
        "string_1d": np.array(["apple", "banana", "cherry"], dtype="S10"),
        "string_2d": np.array([["red", "green"], ["blue", "yellow"]], dtype="S10"),
        # "unicode_1d": np.array(["café", "résumé", "naïve"], dtype="U10")
    }

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Read data back
    with xtype.File(temp_file.name, 'r') as xf:
        read_data = xf.read()

    # Compare original and read data
    for key, value in test_data.items():
        np.testing.assert_array_equal(read_data[key], value)
        assert read_data[key].dtype == value.dtype
        assert read_data[key].shape == value.shape

def test_array_indexing(temp_file):
    """Test array indexing functionality for serialized NumPy arrays."""
    # Create test data
    array_4d = np.arange(120).reshape(2, 3, 4, 5)
    test_data = {"array_4d": array_4d}

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Test various indexing operations
    with xtype.File(temp_file.name, 'r') as xf:
        # Integer indexing
        np.testing.assert_array_equal(xf["array_4d"][0], array_4d[0])
        np.testing.assert_array_equal(xf["array_4d"][1, 2], array_4d[1, 2])
        np.testing.assert_array_equal(xf["array_4d"][1, 2, 3], array_4d[1, 2, 3])
        np.testing.assert_array_equal(xf["array_4d"][1, 2, 3, 4], array_4d[1, 2, 3, 4])

        # Negative indexing
        np.testing.assert_array_equal(xf["array_4d"][-1], array_4d[-1])
        np.testing.assert_array_equal(xf["array_4d"][-1, -1], array_4d[-1, -1])

        # Slice indexing
        np.testing.assert_array_equal(xf["array_4d"][:], array_4d[:])
        np.testing.assert_array_equal(xf["array_4d"][0:1], array_4d[0:1])
        np.testing.assert_array_equal(xf["array_4d"][:, 1:3], array_4d[:, 1:3])
        np.testing.assert_array_equal(xf["array_4d"][0, :, 2:4], array_4d[0, :, 2:4])
        np.testing.assert_array_equal(xf["array_4d"][1, 1, :, :], array_4d[1, 1, :, :])

        # Step slicing
        np.testing.assert_array_equal(xf["array_4d"][::2], array_4d[::2])
        np.testing.assert_array_equal(xf["array_4d"][:, ::2], array_4d[:, ::2])
        np.testing.assert_array_equal(xf["array_4d"][0, 0, ::2], array_4d[0, 0, ::2])

        # List indexing
        np.testing.assert_array_equal(xf["array_4d"][[0]], array_4d[[0]])
        np.testing.assert_array_equal(xf["array_4d"][[0, 1], 1], array_4d[[0, 1], 1])
        np.testing.assert_array_equal(xf["array_4d"][0, [0, 2]], array_4d[0, [0, 2]])

        # Mixed indexing
        np.testing.assert_array_equal(xf["array_4d"][0:2, 1, [0, 2]], array_4d[0:2, 1, [0, 2]])
        np.testing.assert_array_equal(xf["array_4d"][[1], :2, 0], array_4d[[1], :2, 0])

def test_array_edge_cases(temp_file):
    """Test edge cases of array indexing."""
    # Create test data
    array_3d = np.arange(60).reshape(3, 4, 5)
    test_data = {"array_3d": array_3d}

    # Write data to file
    with xtype.File(temp_file.name, 'w') as xf:
        xf.write(test_data)

    # Test edge cases
    with xtype.File(temp_file.name, 'r') as xf:
        # Empty slice
        np.testing.assert_array_equal(xf["array_3d"][3:4], array_3d[3:4])  # Outside bounds
        np.testing.assert_array_equal(xf["array_3d"][2:1], array_3d[2:1])  # Inverse bounds

        # Empty dimensions from list indexing
        np.testing.assert_array_equal(xf["array_3d"][[], :, :], array_3d[[], :, :])

        # Slice with step
        np.testing.assert_array_equal(xf["array_3d"][::2, :, :], array_3d[::2, :, :])

        # Slice with negative step
        np.testing.assert_array_equal(xf["array_3d"][::-1, :, :], array_3d[::-1, :, :])

        # Single-element slice that produces a scalar
        np.testing.assert_array_equal(xf["array_3d"][0, 0, 0], array_3d[0, 0, 0])

        # Single element array slice
        np.testing.assert_array_equal(xf["array_3d"][0:1, 0:1, 0:1], array_3d[0:1, 0:1, 0:1])

        # Complex slicing pattern
        np.testing.assert_array_equal(xf["array_3d"][::2, 1:3, ::-1], array_3d[::2, 1:3, ::-1])
        
        # Index errors
        with pytest.raises(IndexError):
            xf["array_3d"][3]  # Index out of bounds

        with pytest.raises(IndexError):
            xf["array_3d"][0, 0, 5]  # Nested index out of bounds

        # Type errors
        with pytest.raises(TypeError):
            xf["array_3d"]["invalid"]  # Invalid index type

def test_array_setitem_single_element(temp_file):
    """Test setting individual elements in arrays using __setitem__."""
    # Create a test file
    original_data = create_test_file_with_arrays(temp_file)

    with xtype.File(temp_file.name, 'a') as xf:
        # Modify individual elements in 1D array
        xf["array_1d"][2] = 50
        xf["array_1d"][5] = 51

        # Modify individual elements in 2D array
        xf["array_2d"][0, 1] = 25.5
        xf["array_2d"][1, 2] = 26.5

        # Modify individual elements in 3D array
        xf["array_3d"][0, 1, 2] = 111
        xf["array_3d"][1, 2, 3] = 222

        # Modify individual elements in 4D array
        xf["array_4d"][0, 1, 2, 3] = 60
        xf["array_4d"][1, 2, 3, 4] = 61

    # Verify the changes
    with xtype.File(temp_file.name, 'r') as xf:
        # Check 1D array modifications
        np.testing.assert_array_equal(xf["array_1d"][2], 50)
        np.testing.assert_array_equal(xf["array_1d"][5], 51)

        # Check 2D array modifications
        np.testing.assert_array_equal(xf["array_2d"][0, 1], 25.5)
        np.testing.assert_array_equal(xf["array_2d"][1, 2], 26.5)

        # Check 3D array modifications
        np.testing.assert_array_equal(xf["array_3d"][0, 1, 2], 111)
        np.testing.assert_array_equal(xf["array_3d"][1, 2, 3], 222)

        # Check 4D array modifications
        np.testing.assert_array_equal(xf["array_4d"][0, 1, 2, 3], 60)
        np.testing.assert_array_equal(xf["array_4d"][1, 2, 3, 4], 61)

def test_array_setitem_slices(temp_file):
    """Test setting slices of arrays using __setitem__."""
    create_test_file_with_arrays(temp_file)

    with xtype.File(temp_file.name, 'a') as xf:
        # Modify slices in 1D array
        xf["array_1d"][2:5] = np.array([50, 51, 52])
        xf["array_1d"][:2] = np.array([90, 91])

        # Modify slices in 2D array
        xf["array_2d"][0, :] = np.array([25.5, 26.5, 27.5, 28.5], dtype=np.float32)
        xf["array_2d"][1:, 2] = np.array([31.5, 32.5], dtype=np.float32)  # Fixed overlapping indices

        # Modify slices in 3D array
        xf["array_3d"][0, 1, :] = np.array([111, 222, 333, 444], dtype=np.int16)
        xf["array_3d"][1, :, 2] = np.array([555, 666, 777], dtype=np.int16)

        # Modify slices in 4D array
        xf["array_4d"][0, 1, 2, :] = np.array([60, 61, 62, 63, 64], dtype=np.int8)
        xf["array_4d"][1, :, 0, 1] = np.array([70, 71, 72], dtype=np.int8)

    # Verify the changes
    with xtype.File(temp_file.name, 'r') as xf:
        # Check 1D array modifications
        np.testing.assert_array_equal(xf["array_1d"][2:5], np.array([50, 51, 52]))
        np.testing.assert_array_equal(xf["array_1d"][:2], np.array([90, 91]))

        # Check 2D array modifications
        np.testing.assert_array_equal(xf["array_2d"][0, :], np.array([25.5, 26.5, 27.5, 28.5], dtype=np.float32))
        np.testing.assert_array_equal(xf["array_2d"][1:, 2], np.array([31.5, 32.5], dtype=np.float32))  # Fixed verification

        # Check 3D array modifications
        np.testing.assert_array_equal(xf["array_3d"][0, 1, :], np.array([111, 222, 333, 444], dtype=np.int16))
        np.testing.assert_array_equal(xf["array_3d"][1, :, 2], np.array([555, 666, 777], dtype=np.int16))

        # Check 4D array modifications
        np.testing.assert_array_equal(xf["array_4d"][0, 1, 2, :], np.array([60, 61, 62, 63, 64], dtype=np.int8))
        np.testing.assert_array_equal(xf["array_4d"][1, :, 0, 1], np.array([70, 71, 72], dtype=np.int8))

def test_array_setitem_scalar_broadcast(temp_file):
    """Test setting scalar values to array sections (broadcasting)."""
    create_test_file_with_arrays(temp_file)

    with xtype.File(temp_file.name, 'a') as xf:
        # Broadcast to 1D slices
        xf["array_1d"][3:7] = 42

        # Broadcast to 2D slices
        xf["array_2d"][1:3, 1:3] = 99.5

        # Broadcast to 3D slices
        xf["array_3d"][0, :, :] = 123

        # Broadcast to 4D slices
        xf["array_4d"][:, :, 2, :] = 77

    # Verify the changes
    with xtype.File(temp_file.name, 'r') as xf:
        # Check 1D array broadcast
        np.testing.assert_array_equal(xf["array_1d"][3:7], np.full(4, 42))

        # Check 2D array broadcast
        np.testing.assert_array_equal(xf["array_2d"][1:3, 1:3], np.full((2, 2), 99.5))

        # Check 3D array broadcast
        np.testing.assert_array_equal(xf["array_3d"][0, :, :], np.full((3, 4), 123))

        # Check 4D array broadcast
        np.testing.assert_array_equal(xf["array_4d"][:, :, 2, :], np.full((2, 3, 5), 77))

def test_array_setitem_advanced_indexing(temp_file):
    """Test setting values with more complex indexing."""
    create_test_file_with_arrays(temp_file)

    with xtype.File(temp_file.name, 'a') as xf:
        # Set non-contiguous elements in 1D array
        xf["array_1d"][[0, 2, 5, 8]] = np.array([100, 101, 102, 103])

        # Test assigning the same value to multiple indices
        xf["array_1d"][[1, 3, 7, 9]] = 55

        # Mixed indexing in 2D arrays
        xf["array_2d"][:2, [0, 2]] = np.array([[11.1, 22.2], [33.3, 44.4]], dtype=np.float32)

        # Negative indexing
        xf["array_3d"][0, -1, 1:-1] = np.array([500, 501], dtype=np.int16)

    # Verify the changes
    with xtype.File(temp_file.name, 'r') as xf:
        # Check non-contiguous 1D indexing
        np.testing.assert_array_equal(xf["array_1d"][[0, 2, 5, 8]], np.array([100, 101, 102, 103]))

        # Check broadcast to multiple indices
        np.testing.assert_array_equal(xf["array_1d"][[1, 3, 7, 9]], np.array([55, 55, 55, 55]))

        # Check mixed indexing
        np.testing.assert_approx_equal(np.sum(np.abs(xf["array_2d"][:2, [0, 2]])), np.sum(np.abs(np.array([[11.1, 22.2], [33.3, 44.4]]))))

        # Check negative indexing
        np.testing.assert_array_equal(xf["array_3d"][0, -1, 1:-1], np.array([500, 501]))

def test_array_setitem_error_cases(temp_file):
    """Test error conditions for array __setitem__ operations."""
    create_test_file_with_arrays(temp_file)

    with xtype.File(temp_file.name, 'a') as xf:
        # Shape mismatch
        with pytest.raises(ValueError):
            xf["array_1d"][2:5] = np.array([1, 2])  # Wrong number of elements

        # Dtype mismatch (int array to float array)
        with pytest.raises(ValueError):
            # Try to assign int64 array to float32 array
            xf["array_2d"][1, :] = np.array([1, 2, 3, 4], dtype=np.int64)

        # Index out of bounds
        with pytest.raises(IndexError):
            xf["array_1d"][20] = 42

        # Too many indices
        with pytest.raises(IndexError):
            xf["array_2d"][1, 2, 3] = 100
