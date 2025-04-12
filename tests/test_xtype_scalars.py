"""
Unit tests for xtype library - NumPy scalar functionality tests

This test file focuses on NumPy scalar serialization and deserialization.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Add the lib directory to the path to import xtype
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lib")
import xtype


class TestXTypeScalars(unittest.TestCase):
    """Test NumPy scalar functionality of the xtype library."""

    def setUp(self):
        """Set up a temporary file for tests."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary files after tests."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_numpy_scalars(self):
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
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Compare original and read data
        for key, value in test_data.items():
            if key != "mixed_data":
                self.assertEqual(read_data[key], value)
                # self.assertEqual(type(read_data[key]), type(value))
            else:
                # Handle nested dictionary case
                np.testing.assert_array_equal(read_data[key]["array"], value["array"])
                self.assertEqual(read_data[key]["scalar"], value["scalar"])
                # self.assertEqual(type(read_data[key]["scalar"]), type(value["scalar"]))

    def test_scalar_conversion(self):
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
        with xtype.File(self.temp_file.name, 'w') as xf:
            xf.write(test_data)

        # Read data back
        with xtype.File(self.temp_file.name, 'r') as xf:
            read_data = xf.read()

        # Verify that the keys were converted to strings (as required by xtype)
        self.assertEqual(read_data["dict_with_numpy_keys"]["1"], "value1")
        self.assertEqual(read_data["dict_with_numpy_keys"]["2"], "value2")

        # Verify that the list values match (may be Python native types on read)
        self.assertEqual(read_data["numpy_scalar_in_list"][0], 1)
        self.assertEqual(read_data["numpy_scalar_in_list"][1], 2.5)
        self.assertEqual(read_data["numpy_scalar_in_list"][2], True)


if __name__ == '__main__':
    unittest.main()
