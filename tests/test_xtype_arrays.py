"""
Unit tests for xtype library - NumPy array functionality tests

This test file focuses on NumPy array serialization, deserialization,
and the array indexing functionality including multi-dimensional arrays.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Add the lib directory to the path to import xtype
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lib")
import xtype


class TestXTypeArrays(unittest.TestCase):
    """Test NumPy array functionality of the xtype library."""

    def setUp(self):
        """Set up a temporary file for tests."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary files after tests."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_1d_arrays(self):
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
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        for key, value in test_data.items():
            np.testing.assert_array_equal(read_data[key], value)
            self.assertEqual(read_data[key].dtype, value.dtype)

    def test_2d_arrays(self):
        """Test serializing and deserializing 2D NumPy arrays."""
        test_data = {
            "int32_2d": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            "float64_2d": np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float64),
            "bool_2d": np.array([[True, False], [False, True]], dtype=bool)
        }

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        for key, value in test_data.items():
            np.testing.assert_array_equal(read_data[key], value)
            self.assertEqual(read_data[key].dtype, value.dtype)
            self.assertEqual(read_data[key].shape, value.shape)

    def test_multidimensional_arrays(self):
        """Test serializing and deserializing multi-dimensional NumPy arrays."""
        test_data = {
            "3d_array": np.arange(60).reshape(3, 4, 5),
            "4d_array": np.arange(120).reshape(2, 3, 4, 5)
        }

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        for key, value in test_data.items():
            np.testing.assert_array_equal(read_data[key], value)
            self.assertEqual(read_data[key].dtype, value.dtype)
            self.assertEqual(read_data[key].shape, value.shape)

    def test_string_arrays(self):
        """Test serializing and deserializing string arrays."""
        test_data = {
            "string_1d": np.array(["apple", "banana", "cherry"], dtype="S10"),
            "string_2d": np.array([["red", "green"], ["blue", "yellow"]], dtype="S10"),
            # "unicode_1d": np.array(["café", "résumé", "naïve"], dtype="U10")
        }

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        for key, value in test_data.items():
            np.testing.assert_array_equal(read_data[key], value)
            self.assertEqual(read_data[key].dtype, value.dtype)
            self.assertEqual(read_data[key].shape, value.shape)

    def test_array_indexing(self):
        """Test array indexing functionality for serialized NumPy arrays."""
        # Create test data
        array_4d = np.arange(120).reshape(2, 3, 4, 5)
        test_data = {"array_4d": array_4d}

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Test various indexing operations
        with xtype.File(self.temp_file.name, 'r') as xf:
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

    def test_array_edge_cases(self):
        """Test edge cases of array indexing."""
        # Create test data
        array_3d = np.arange(60).reshape(3, 4, 5)
        test_data = {"array_3d": array_3d}

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Test edge cases
        with xtype.File(self.temp_file.name, 'r') as xf:
            # Empty slice
            np.testing.assert_array_equal(xf["array_3d"][3:4], array_3d[3:4])  # Outside bounds
            np.testing.assert_array_equal(xf["array_3d"][2:1], array_3d[2:1])  # Inverse bounds

            # Empty dimensions from list indexing
            np.testing.assert_array_equal(xf["array_3d"][[], :, :], array_3d[[], :, :])

            # Index errors
            with self.assertRaises(IndexError):
                xf["array_3d"][3]  # Index out of bounds

            with self.assertRaises(IndexError):
                xf["array_3d"][0, 0, 5]  # Nested index out of bounds

            # Type errors
            with self.assertRaises(TypeError):
                xf["array_3d"]["invalid"]  # Invalid index type


if __name__ == '__main__':
    unittest.main()
