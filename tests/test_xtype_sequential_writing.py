import os
import sys
import tempfile
import pytest
import numpy as np

# Add library directory to path to import xtype
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))
import xtype

def test_sequential_writing(tmp_path):
    """
    Test the alternative approach to sequential writing using tuples and dicts.
    """
    test_file = tmp_path / "test_alternative_sequential.xtype"

    # Create the same data structure using the alternative approach
    with xtype.File(test_file, 'w') as xf:
        xf["a"] = 5                             # "a": 5,
        xf["list"] = []                         # "list":
        list1 = xf.last                         #
        list2 = list1.add([])                   #
        list2.add(1)                            #     1
        list2.add(4)                            #     4
        dict1 = list2.add({})                   #
        dict1['five'] = 5                       #        'five': 5
        xf.last['six'] = 6                      #        'six': 6
        list1.add(7)                            #     7
        xf['dict'] = {}                         #  "dict":
        xf.last['numbers'] = []                 #     "numbers":
        xf.last.add(1)                          #                1
        xf.last.add(2)                          #                2
        xf.last.add(3)                          #                3

    # Read back the data and verify it matches the expected structure
    with xtype.File(str(test_file), 'r') as xf:
        data = xf.read()

    expected = {
        "a": 5,
        "list": [
            [1, 4, {"five": 5, "six": 6}],
            7
        ],
        "dict": {
            "numbers": [1, 2, 3]
        }
    }
    print('expected:', expected)
    print('read:    ', data)
    assert data == expected

def test_sequential_write_closed_file(tmp_path):
    """
    Test that writing to a closed xtype.File raises an exception.
    """
    test_file = tmp_path / "test_closed.xtype"
    xf = xtype.File(test_file, 'w')
    xf.close()
    with pytest.raises(ValueError):
        xf["a"] = 1

def test_sequential_write_unsupported_type(tmp_path):
    """
    Test that writing an unsupported type raises an exception.
    """
    test_file = tmp_path / "test_unsupported.xtype"
    with xtype.File(test_file, 'w') as xf:
        class Dummy:
            pass
        with pytest.raises(TypeError):
            xf["dummy"] = Dummy()

def test_sequential_write_add_to_nonlist(tmp_path):
    """
    Test that calling add() on a non-list/non-dict raises an exception.
    """
    test_file = tmp_path / "test_nonlist.xtype"
    with xtype.File(test_file, 'w') as xf:
        xf["a"] = 5
        with pytest.raises(TypeError):
            xf.last.add(1)

def test_sequential_write_large_nested(tmp_path):
    """
    Test writing a large nested structure to check for stack/recursion errors.
    """
    test_file = tmp_path / "test_large_nested.xtype"
    with xtype.File(test_file, 'w') as xf:
        current = xf
        for i in range(100):
            current["a"] = {}
            current = xf.last
        # Should not raise
    with xtype.File(str(test_file), 'r') as xf:
        data = xf.read()
        cur = data
        for i in range(100):
            assert "a" in cur
            cur = cur["a"]
