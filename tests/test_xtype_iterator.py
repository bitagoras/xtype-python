"""
Unit tests for xtype library - Iterator functionality tests

This test file focuses on testing the iterator functionality of
the readPointer class, including simple lists, complex nested structures,
and error handling for non-iterable objects.
"""
import os
import sys
import tempfile
import pytest
import numpy as np

# Add library directory to path to import xtype
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))
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

def test_list_iterator(temp_file):
    """Test that the readPointer iterator works correctly with a simple list."""
    test_data = [1, 2, 3, 4, 5]

    # Write test data
    with xtype.File(temp_file.name, 'w') as f:
        f.write(test_data)

    # Read data using iterator
    with xtype.File(temp_file.name, 'r') as f:
        # Convert iterator results to a list
        result = list(f.root)

        # Verify result matches original
        assert result == test_data

        # Test that we can iterate multiple times
        second_result = list(f.root)
        assert second_result == test_data

        # Check that indexing still works after iteration
        assert f.root[2] == 3


def test_complex_list_iterator(temp_file):
    """Test iterator with a more complex list containing nested structures."""
    test_data = [
        {"name": "item1", "value": 10},
        {"name": "item2", "value": 20},
        [1, 2, 3],
        np.array([4, 5, 6])
    ]

    # Write test data
    with xtype.File(temp_file.name, 'w') as f:
        f.write(test_data)

    # Read data using iterator
    with xtype.File(temp_file.name, 'r') as f:
        # Get items one by one
        items = []
        for item in f:
            if isinstance(item, dict):
                # Convert keys to list to ensure consistent order
                assert sorted(item.keys()) == ["name", "value"]
                if item["name"] == "item1":
                    assert item["value"] == 10
                elif item["name"] == "item2":
                    assert item["value"] == 20
            elif isinstance(item, list):
                assert item == [1, 2, 3]
            elif isinstance(item, np.ndarray):
                np.testing.assert_array_equal(item, np.array([4, 5, 6]))

            items.append(item)

        # Verify we got the right number of items
        assert len(items) == 4


def test_non_list_iterator(temp_file):
    """Test that attempting to iterate over a non-list raises TypeError."""
    test_data = {"key": "value"}

    # Write test data
    with xtype.File(temp_file.name, 'w') as f:
        f.write(test_data)

    # Try to iterate a dictionary
    with xtype.File(temp_file.name, 'r') as f:
        root_pointer = f.root

        # Should raise TypeError
        with pytest.raises(TypeError):
            list(root_pointer)
