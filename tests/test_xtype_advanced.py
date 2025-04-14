"""
Unit tests for xtype library - Advanced features and edge cases

This test file covers more advanced features like byteorder handling,
large file operations, and other edge cases.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import struct

# Add the lib directory to the path to import xtype
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lib")
import xtype


class TestXTypeAdvanced(unittest.TestCase):
    """Test advanced features and edge cases of the xtype library."""

    def setUp(self):
        """Set up a temporary file for tests."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary files after tests."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_byteorder_big(self):
        """Test explicit big-endian byte order."""
        test_data = {
            "int32": np.array([123456789], dtype=np.int32)[0],
            "float64": 3.14159265359,
            "array": np.array([1.0, 2.0, 3.0], dtype=np.float64)
        }

        # Write data with big-endian byte order
        with xtype.File(self.temp_file.name, 'w', byteorder='big') as xf:
            xf.write(test_data)

        # Read data back with auto byte order detection
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        self.assertEqual(read_data["int32"], test_data["int32"])
        self.assertEqual(read_data["float64"], test_data["float64"])
        np.testing.assert_array_equal(read_data["array"], test_data["array"])

    def test_byteorder_little(self):
        """Test explicit little-endian byte order."""
        test_data = {
            "int32": np.array([123456789], dtype=np.int32)[0],
            "float64": 3.14159265359,
            "array": np.array([1.0, 2.0, 3.0], dtype=np.float64)
        }

        # Write data with little-endian byte order
        with xtype.File(self.temp_file.name, 'w', byteorder='little') as xf:
            xf.write(test_data)

        # Read data back with auto byte order detection
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        self.assertEqual(read_data["int32"], test_data["int32"])
        self.assertEqual(read_data["float64"], test_data["float64"])
        np.testing.assert_array_equal(read_data["array"], test_data["array"])

    def test_large_integers(self):
        """Test serializing and deserializing very large integers."""
        test_data = {
            "small_int": 42,
            "medium_int": 123456789,
            "large_int": 9223372036854775807,  # 2^63 - 1 (max int64)
            "very_large_int": 18446744073709551615  # 2^64 - 1 (max uint64)
        }

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        self.assertEqual(read_data, test_data)

    def test_empty_file(self):
        """Test handling of empty files."""
        # Create an empty file
        with open(self.temp_file.name, 'wb') as f:
            pass

        # Try to read from empty file
        with xtype.File(self.temp_file.name, 'r') as xf:
            self.assertIsNone(xf)  # Empty file should return None

    def test_nested_structure_depth(self):
        """Test deeply nested data structures."""
        # Create a deeply nested dictionary
        test_data = {}
        current = test_data
        for i in range(10):  # 10 levels of nesting
            current["level"] = {}
            current = current["level"]
        current["value"] = 42

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        self.assertEqual(read_data, test_data)

        # Test direct access to deeply nested value
        with xtype.File(self.temp_file.name, 'r') as xf:
            # Build path to deeply nested value
            obj = xf
            for i in range(10):
                obj = obj["level"]
            self.assertEqual(obj["value"], 42)

    def test_large_array(self):
        """Test serializing and deserializing a large array."""
        # Create a large array (1 million elements)
        large_array = np.arange(1_000_000, dtype=np.int32)
        test_data = {"large_array": large_array}

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back using direct indexing without loading the entire array
        with xtype.File(self.temp_file.name, 'r') as xf:
            # Check size without loading array
            self.assertEqual(len(xf["large_array"]), 1_000_000)

            # Test random access to elements
            self.assertEqual(xf["large_array"][0], 0)
            self.assertEqual(xf["large_array"][999_999], 999_999)
            self.assertEqual(xf["large_array"][500_000], 500_000)

            # Test slicing on large array
            np.testing.assert_array_equal(
                xf["large_array"][500_000:500_010],
                large_array[500_000:500_010]
            )

            # Test strided slicing
            np.testing.assert_array_equal(
                xf["large_array"][500_000:500_100:10],
                large_array[500_000:500_100:10]
            )

    def test_file_operations(self):
        """Test file operations and context management."""
        test_data = {"test": 123}

        # Test context manager
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Test manual open/close
        xf = xtype.File(self.temp_file.name, 'r')
        xf.open()
        self.assertEqual(xf.read(), test_data)
        xf.close()

        # Test file is closed properly
        self.assertTrue(xf.file.closed)

        # Test reusing the same file object
        xf = xtype.File(self.temp_file.name, 'r')
        xf.open()
        result = xf.read()
        xf.close()
        self.assertEqual(result, test_data)

    def test_mixed_complex_structures(self):
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
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        # Need to check each part separately due to NumPy arrays
        # Scalars
        self.assertEqual(read_data["scalars"], test_data["scalars"])

        # Arrays
        for key, value in test_data["arrays"].items():
            np.testing.assert_array_equal(read_data["arrays"][key], value)

        # Mixed list - special handling for NumPy array
        self.assertEqual(read_data["mixed_list"][0], test_data["mixed_list"][0])
        self.assertEqual(read_data["mixed_list"][1], test_data["mixed_list"][1])
        np.testing.assert_array_equal(
            read_data["mixed_list"][2],
            test_data["mixed_list"][2]
        )
        self.assertEqual(read_data["mixed_list"][3], test_data["mixed_list"][3])

    def test_keys_method(self):
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
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read back and test keys method
        with xtype.File(self.temp_file.name, 'r') as xf:
            # Test root keys
            self.assertEqual(set(xf.keys()), {"a", "b", "c"})

            # Test nested keys
            self.assertEqual(set(xf["c"].keys()), {"d", "e"})

    def test_len_method(self):
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
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read back and test len method
        with xtype.File(self.temp_file.name, 'r') as xf:
            # Test dictionary lengths
            self.assertEqual(len(xf["dict"]), 3)
            self.assertEqual(len(xf["nested"]["inner_dict"]), 2)

            # Test list lengths
            self.assertEqual(len(xf["list"]), 5)
            self.assertEqual(len(xf["nested"]["inner_list"]), 3)

            # Test root length (should be 3 keys)
            self.assertEqual(len(xf), 3)

            # Test len() raises TypeError for non-container types
            with self.assertRaises(TypeError):
                len(xf["dict"]["a"])


if __name__ == '__main__':
    unittest.main()
