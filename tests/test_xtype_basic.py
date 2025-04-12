"""
Unit tests for xtype library - Basic functionality tests

This test file covers basic serialization/deserialization of primitive types,
lists, dictionaries, and basic error handling.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Add the lib directory to the path to import xtype
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lib")
import xtype


class TestXTypeBasic(unittest.TestCase):
    """Test basic functionality of the xtype library."""

    def setUp(self):
        """Set up a temporary file for tests."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary files after tests."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_primitive_types(self):
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
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        self.assertEqual(read_data, test_data)

        # Test individual element access
        with xtype.File(self.temp_file.name, 'r') as xf:
            self.assertEqual(xf["integer"](), 42)
            self.assertEqual(xf["negative_int"](), -17)
            self.assertEqual(xf["large_int"](), 9223372036854775807)
            self.assertEqual(xf["float"](), 3.14159265359)
            self.assertTrue(xf["boolean_true"]())
            self.assertFalse(xf["boolean_false"]())
            self.assertIsNone(xf["none_value"]())
            self.assertEqual(xf["string"](), "Hello, world!")
            self.assertEqual(xf["bytes"](), b"Binary data")

    def test_container_types(self):
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
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        self.assertEqual(read_data, test_data)

        # Test list operations
        with xtype.File(self.temp_file.name, 'r') as xf:
            self.assertEqual(len(xf["list_of_ints"]), 5)
            self.assertEqual(xf["list_of_ints"][0](), 1)
            self.assertEqual(xf["list_of_ints"][4](), 5)
            self.assertEqual(xf["nested_list"][1][1](), 3)
            self.assertEqual(xf["nested_list"][2][1][1](), 6)

        # Test dict operations
        with xtype.File(self.temp_file.name, 'r') as xf:
            self.assertEqual(len(xf["simple_dict"]), 3)
            self.assertEqual(xf["simple_dict"]["a"](), 1)
            self.assertEqual(xf["simple_dict"]["c"](), 3)
            self.assertEqual(xf["nested_dict"]["a"]["b"]["c"](), 42)

    def test_slicing(self):
        """Test list slicing operations."""
        test_data = {
            "list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }

        # Write data to file
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Test slicing operations
        with xtype.File(self.temp_file.name, 'r') as xf:
            # Basic slices
            self.assertEqual(xf["list"][2:5], [2, 3, 4])
            self.assertEqual(xf["list"][:3], [0, 1, 2])
            self.assertEqual(xf["list"][7:], [7, 8, 9])
            self.assertEqual(xf["list"][:], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

            # Slices with steps
            self.assertEqual(xf["list"][::2], [0, 2, 4, 6, 8])
            self.assertEqual(xf["list"][1::2], [1, 3, 5, 7, 9])
            self.assertEqual(xf["list"][1:8:3], [1, 4, 7])

    def test_error_cases(self):
        """Test error handling."""
        # Create a simple file with dictionary data
        test_data = {"key1": "value1", "key2": [1, 2, 3]}

        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Test different error cases
        with xtype.File(self.temp_file.name, 'r') as xf:
            # Key not found
            with self.assertRaises(KeyError):
                xf["non_existent_key"]()

            # Index out of range
            with self.assertRaises(IndexError):
                xf["key2"][5]()

            # Type error (trying to index a non-indexable object)
            with self.assertRaises(TypeError):
                xf["key1"][0]()

    def test_file_modes(self):
        """Test file opening in different modes."""
        # Test write mode
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write({"test": 123})

        # Test read mode
        with xtype.File(self.temp_file.name, 'r') as xf:
            self.assertEqual(xf.read(), {"test": 123})

        # Test invalid mode
        with self.assertRaises(ValueError):
            with xtype.File(self.temp_file.name, 'x') as xf:
                pass


if __name__ == '__main__':
    unittest.main()
