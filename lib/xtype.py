"""
xtype - Universal Binary Notation Language for Data Exchange

A Python implementation for serializing and deserializing data structures
using the xtype binary format, optimized for efficient data exchange and storage.
This library provides a compact, efficient binary format for representing
complex data structures including nested containers and multi-dimensional arrays.

Features:
- Serialization of basic types (int, float, str, bytes, bool)
- Support for container types (list, dict)
- Efficient handling of NumPy arrays (1D, 2D, and higher-dimensional)
- Memory-efficient partial loading via object pointers
- Compact binary representation

Copyright (c) 2025 Bitagoras
License: MIT

Project: https://github.com/bitagoras/xtype-python
"""

__version__ = "0.5.1"

import struct
import numpy as np
from typing import Any, Dict, List, Tuple, BinaryIO, Iterator, Optional, Union
import sys
import itertools


# Grammar of xtype format
# This formal grammar defines the structure of the xtype binary format
# and is used to guide the parser implementation.

# <file>       ::= <EOF> | <object> <EOF>
# <object>     ::= <content> | <footnote> <content>
# <footnote>   ::= "*" <content> | "*" <content> <footnote>
# <content>    ::= <element> | <list> | <dict>
# <list>       ::= "[]" | "[" <list_items> "]" | "[" <EOF> | "[" <list_items> <EOF>
# <list_items> ::= <object> | <object> <list_items>
# <dict>       ::= "{}" | "{" <dict_items> "}" | "{" <EOF> | "{" <list_items> <EOF>
# <dict_items> ::= <element> <object> | <element> <object> <dict_items>
# <element>    ::= <type> <bin_data> | "T"  | "F" | "n"
# <type>       ::= <length> <type> | <bin_data>
# <length>     ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" |
#                  "M" <bin_data> | "N" <bin_data> | "O" <bin_data> | "P" <bin_data>
# <bin_type>   ::= "i" | "j" | "k" | "l" | "I" | "J" | "K" | "L" |
#                  "b" | "h" | "f" | "d" | "s" | "u" | "S" | "x"

# <bin_data> represents the actual binary data of the specified size for the given type.
# <EOF> marks the end of file. In streaming applications, this could be represented by a zero byte.

class ContainerProxy:
    "just a test"
    """Abstract base class for all xtype proxies (container, array, etc). Internal base class for container proxies (lists & dicts)."""
    def __init__(self, xtFile, parent, opening_char: bytes, closing_char: bytes):
        self.xtFile = xtFile
        self.parent = parent
        self._closed = False  # For containers, True if closed
        self._closing_char = closing_char  # Byte used to close the container
        # For arrays: additional metadata (shape, dtype, etc) can be set by subclass

        # Basic safety checks – refuse to operate on a closed/non‑writable file
        if getattr(self.xtFile, "file", None) is None or getattr(self.xtFile, "writer", None) is None:
            raise ValueError("Cannot write to file: file is closed.")

        # Emit the opening token for the container immediately
        if opening_char:
            self.xtFile.writer._buffer.append(opening_char)
            self.xtFile.writer.flush()

    # ------------------------------------------------------------------
    # Helper used by both ListProxy.add() and DictProxy.__setitem__()
    # ------------------------------------------------------------------
    def _handle_value(self, value):
        """Write *value* (or open a nested container) and return nested proxy if any."""
        self.xtFile._close_to(self)
        if isinstance(value, dict):
            proxy = DictProxy(self.xtFile, parent=self)
            self.xtFile._open_containers.append(proxy)
            self.xtFile.last = proxy
            self.xtFile.writer.flush()
            return proxy
        elif isinstance(value, list):
            proxy = ListProxy(self.xtFile, parent=self)
            self.xtFile._open_containers.append(proxy)
            self.xtFile.last = proxy
            self.xtFile.writer.flush()
            return proxy
        else:
            self.xtFile.writer._write_object(value)
            self.xtFile.writer.flush()
            self.xtFile.last = self
            return None

    def _ensure_not_closed(self, container_name: str):
        if self._closed:
            raise RuntimeError(f"Cannot write to closed {container_name}.")

    def _close(self):
        """Write the container's closing token if not already closed."""
        if not self._closed:
            self.xtFile.writer._buffer.append(self._closing_char)
            self.xtFile.writer.flush()
            self._closed = True

class ListProxy(ContainerProxy):
    def __init__(self, xtFile, parent):
        # '[' opens list, ']' closes
        super().__init__(xtFile, parent, b'[', b']')

    def add(self, value):
        """Append *value* to the list, returning a proxy for nested containers."""
        self._ensure_not_closed("list")
        return self._handle_value(value)

    def __setitem__(self, key, value):
        raise TypeError("Cannot use __setitem__ on a list container. Use add().")


class DictProxy(ContainerProxy):
    def __init__(self, xtFile, parent):
        # '{' opens dict, '}' closes
        super().__init__(xtFile, parent, b'{', b'}')

    def __setitem__(self, key, value):
        self._ensure_not_closed("dict")
        self.xtFile.writer._write_object(key)
        return self._handle_value(value)

    def add(self, value):
        raise TypeError("Cannot use add() on a dict container. Use __setitem__().")


class EmptyFile(Exception):
    """Exception raised when trying to read from an empty file."""

class File:
    def _check_open_for_reading(self):
        if self.file is None:
            if getattr(self, '_was_closed', False):
                raise IOError("File was closed and cannot be reopened.")
            self._open()
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")
        if self.mode not in 'ra':
            raise IOError("File is not open in read mode")

    def _check_open_for_writing(self):
        if self.file is None:
            if getattr(self, '_was_closed', False):
                raise IOError("File was closed and cannot be reopened.")
            self._open()
        if not self.file or self.file.closed:
            raise IOError("File is not open for writing")
        if self.mode != 'w':
            raise IOError("File is not open in write mode")

    """
    A class for reading and writing Python data structures to files using the xtype binary format.

    Supports serialization of:
    - Basic types: int, float, str, bytes, bool
    - Container types: list, dict
    - NumPy arrays: 1D, 2D, and higher-dimensional arrays with various data types

    This class provides a high-level interface for working with xtype files, handling
    both reading and writing operations with access to subelements.
    """

    def __init__(self, filename: str, mode: str = 'r', byteorder: str = 'auto'):
        """
        Initialize an xtype.File object.

        Args:
            filename: Path to the file
            mode: File mode ('w' for write, 'r' for read, 'a' for append/read+write)
            byteorder: The byte order of multi-byte integers in the file.
                       'big', 'little' or 'auto'. Defaults to 'auto'.
                       'auto' selects the systems byte order for writing
                       and 'big' or that of the optional BOM for reading.
        """
        self.filename = filename
        self.mode = mode
        self.file = None

        # Reader and writer instances (initialized in open())
        self.reader = None
        self.writer = None
        self.byteorder = byteorder
        self.root = None
        self.last = None  # Points to the most recently active container proxy
        self._open_containers = []  # Stack of open proxies (root to most nested)
        self._was_closed = False

    def __enter__(self):
        """Context manager entry point."""
        try:
            self._open()
        except EmptyFile:
            return None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        # Close all open containers (from innermost to outermost)
        if self.mode == 'w' and self._open_containers:
            for proxy in reversed(self._open_containers):
                if hasattr(proxy, '_close'):
                    proxy._close()
            self._open_containers.clear()
        self.close()

    def _open(self):
        """Open the file for reading or writing."""
        if self.mode == 'w':
            self.file = open(self.filename, 'wb')
            self.writer = XTypeFileWriter(self, byteorder=self.byteorder)
            self.root = None
            self.last = None
            self._open_containers = []
        elif self.mode == 'r':
            self.file = open(self.filename, 'rb')
            self.reader = XTypeFileReader(self, byteorder=self.byteorder)
        elif self.mode == 'a':
            self.file = open(self.filename, 'r+b')
            self.reader = XTypeFileReader(self, byteorder=self.byteorder)
            self.writer = XTypeFileWriter(self, byteorder=self.byteorder)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.mode in 'ra':
            self.root = ObjectProxy(self, 0)

    def close(self):
        """Close the file."""
        if self.file is None and not self._was_closed:
            self._open()
        if self.file:
            self.file.close()
            self.file = None
            self._was_closed = True

    def __setitem__(self, key, value):
        """
        Incrementally write a key-value pair to the root dict, or create the root dict if not present.
        """
        self._check_open_for_writing()
        if self.root is None:
            # First operation: initialize root as dict
            self.root = DictProxy(self, parent=None)
            self._open_containers.append(self.root)
            self.last = self.root
        elif not isinstance(self.root, DictProxy):
            raise TypeError("Root container is already a list; cannot use dict-style assignment.")
        # Ensure correct container closure
        self._close_to(self.root)
        self.last = self.root
        self.root[key] = value

    def add(self, value):
        """
        Incrementally add a value to the root list, or create the root list if not present.
        """
        self._check_open_for_writing()
        if self.root is None:
            # First operation: initialize root as list
            self.root = ListProxy(self, parent=None)
            self._open_containers.append(self.root)
            self.last = self.root
        elif not isinstance(self.root, ListProxy):
            raise TypeError("Root container is already a dict; cannot use list-style add.")
        # Ensure correct container closure
        self._close_to(self.root)
        self.last = self.root
        self.root.add(value)

    def _close_to(self, target_proxy):
        """
        Close all open containers above and including the top, until target_proxy is on top.
        """
        while self._open_containers and self._open_containers[-1] is not target_proxy:
            proxy = self._open_containers.pop()
            if hasattr(proxy, '_close'):
                proxy._close()
        # Don't pop target_proxy itself

    def write(self, data: Any):
        """
        Write a Python object to the file in xtype format.

        Args:
            data: The Python object to serialize (can be a primitive type,
                 list, dict, or numpy array)
        """
        self._check_open_for_writing()
        self.writer._write_bom()
        self.writer._write_object(data)
        self.writer.flush()
        # If any containers were open, close them
        # self._close_open_containers()


    def read(self) -> Any:
        """
        Read an xtype file and convert it to a Python object.

        Returns:
            Any: The Python object read from the file
        """
        self._check_open_for_reading()
        # Reset the file position to the beginning
        self.file.seek(0)
        # Start recursive parsing
        return self.reader.read()

    def __getitem__(self, key):
        """
        Access an element within the file.

        Creates an ObjectProxy at the beginning of the file and uses its __getitem__ method.
        Supports various indexing operations for different data structures:

        Args:
            key: The index specifier, which can be:
                - Integer index (for lists and arrays)
                - Dictionary key (for dictionaries)
                - Slice object (for lists and arrays)
                - List/array of indices (for arrays)
                - Tuple of indices/slices/lists (for multi-dimensional arrays)

        Returns:
            For lists: An ObjectProxy pointing to the found object (for integer indices)
                      or a Python list (for slices)
            For dictionaries: An ObjectProxy pointing to the found object
            For arrays: The actual array data (NumPy array) or scalar value

        Raises:
            IndexError: If the index is out of bounds or the item is not found
            KeyError: If the dictionary key is not found
            TypeError: If the object does not support the requested indexing operation
            IOError: If the file is not open for reading or not in read mode
        """
        self._check_open_for_reading()
        # Use the ObjectProxy's __getitem__ method
        return self.root[key]

    def read_debug(self, indent_size: int = 2, max_indent_level: int = 10, max_binary_bytes: int = 15) -> Iterator[str]:
        """
        Iterator to read raw data from an xtype file and convert each output to a formatted string.

        This is a convenience method that delegates to the reader's read_debug method.
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        # Reset the file position at the start
        self.file.seek(0)

        return self.reader.read_debug(indent_size, max_indent_level, max_binary_bytes)

    def keys(self):
        """
        Return a list of keys if the root object is a dictionary.

        Returns:
            list: List of keys from the dictionary

        Raises:
            TypeError: If the root object is not a dictionary
            IOError: If the file is not open for reading
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        return self.root.keys()

    def __len__(self):
        """
        Return the length of the root object if it is a list or dictionary.

        Returns:
            int: Number of items in the list or dictionary

        Raises:
            TypeError: If the root object is neither a list nor a dictionary
            IOError: If the file is not open for reading
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        return len(self.root)

    def __iter__(self):
        """
        Enable iteration over a File object by delegating to ObjectProxy.__iter__.

        Returns:
            iterator: An iterator over the elements in the file

        Raises:
            TypeError: If the root object is not a list
            IOError: If the file is not open for reading or not in read mode
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode not in 'ra':
            raise IOError("File is not open in read mode")

        # Delegate to the ObjectProxy's __iter__ method
        return iter(self.root)

class XTypeFileWriter:
    """
    A class for writing Python data structures to files using the xtype binary format.
    """

    # Type mapping between Python/NumPy types and xtype format type codes
    # Used during serialization to determine the appropriate type code for each value
    type_map = {
        # Python native type mappings
        bool: 'b',    # Boolean to xtype boolean
        int: 'k',     # Python int defaults to 32-bit integer
        float: 'd',   # Python float defaults to 64-bit double
        str: 's',     # Python string to UTF-8 encoded string
        bytes: 'x',   # Python bytes to raw byte array

        # NumPy specific type mappings for precise control
        np.dtype('bool'): 'b',    # Boolean
        np.dtype('int8'): 'i',    # 8-bit signed integer
        np.dtype('int16'): 'j',   # 16-bit signed integer
        np.dtype('int32'): 'k',   # 32-bit signed integer
        np.dtype('int64'): 'l',   # 64-bit signed integer
        np.dtype('uint8'): 'I',   # 8-bit unsigned integer
        np.dtype('uint16'): 'J',  # 16-bit unsigned integer
        np.dtype('uint32'): 'K',  # 32-bit unsigned integer
        np.dtype('uint64'): 'L',  # 64-bit unsigned integer
        np.dtype('float16'): 'h', # 16-bit half-precision float
        np.dtype('float32'): 'f', # 32-bit single-precision float
        np.dtype('float64'): 'd'  # 64-bit double-precision float
    }

    def __init__(self, xtFile: File, byteorder: str = 'auto'):
        """
        Initialize an XTypeFileWriter object.

        Args:
            file: The file object to write to
        """
        self.file = xtFile.file
        self.byteorder = byteorder if byteorder != 'auto' else sys.byteorder
        self.need_byteswap = self.byteorder != sys.byteorder
        self.struct_byteorder = {'little': '<', 'big': '>'}[self.byteorder]
        self._buffer = []  # Buffer for binary fragments

    def flush(self):
        """
        Write all buffered binary fragments to the file and clear the buffer.
        """
        for frag in self._buffer:
            self.file.write(frag)
        self._buffer.clear()

    def _write_bom(self):
        """
        Write a byte order mark (BOM) to the file.

        The BOM is a 16-bit signed integer with the value 1234. It is used to indicate the byte order of
        multi-byte integers in the file.

        The BOM is written as a 16-bit signed integer in big-endian byte order. The 16-bit signed integer
        has the defined value of 1234. An xtype reader with the wrong byte order would read the number as
        -11772. If no such file signature is given, xtype is specified for big endian byte order as
        default.
        """
        self._buffer.append(b'*j')
        self._write_int_value(1234, 'j')

    def _write_object(self, obj: Any):
        """
        Write an object to the file.

        Args:
            obj: The object to write
        """
        if isinstance(obj, (list, tuple)):
            self._write_list(obj)
        elif isinstance(obj, dict):
            self._write_dict(obj)
        elif isinstance(obj, np.ndarray):
            self._write_numpy_array(obj)
        elif isinstance(obj, bytes):
            # Handle bytes directly instead of treating as an element
            # This is important for binary data handling
            self._write_element(obj)
        elif obj is None:
            # Handle None explicitly
            self._buffer.append(b'n')
        else:
            self._write_element(obj)

    def _write_list(self, lst: List):
        """
        Write a list to the file.

        Args:
            lst: The list to write
        """
        self._buffer.append(b'[')
        for item in lst:
            self._write_object(item)
        self._buffer.append(b']')

    def _write_dict(self, d: Dict):
        """
        Write a dictionary to the file.

        Args:
            d: The dictionary to write
        """
        self._buffer.append(b'{')
        for key, value in d.items():
            # Convert key to string if it's not already
            if not isinstance(key, str):
                key = str(key)
            # Write the key as a string element
            self._write_element(key)
            # Write the value
            self._write_object(value)
        self._buffer.append(b'}')

    def _write_element(self, value: Any):
        """
        Write a basic element to the file.

        Args:
            value: The value to write
        """
        if value is None:
            self._buffer.append(b'n')
        elif isinstance(value, bool):
            self._buffer.append(b'T' if value else b'F')
        elif isinstance(value, int):
            type_code = self._select_int_type(value)
            self._buffer.append(type_code.encode())
            self._write_int_value(value, type_code)
        elif isinstance(value, float):
            self._buffer.append(b'd')
            self._buffer.append(struct.pack(f'{self.struct_byteorder}d', value))
        elif isinstance(value, str):
            # Write string with length prefix
            encoded = value.encode('utf-8')
            self._write_length(len(encoded))
            self._buffer.append(b's')
            self._buffer.append(encoded)
        elif isinstance(value, bytes):
            # Write bytes with length prefix
            self._write_length(len(value))
            self._buffer.append(b'x')
            self._buffer.append(value)
        elif isinstance(value, np.number) or isinstance(value, np.bool_):
            # Handle NumPy scalar types
            dtype = value.dtype
            if dtype in self.type_map:
                type_code = self.type_map[dtype]
                self._buffer.append(type_code.encode())

                # Process based on the specific scalar type
                if np.issubdtype(dtype, np.integer):
                    # Handle integer types
                    if np.issubdtype(dtype, np.integer):
                        data = np.asarray(value, dtype=dtype)
                        # Only need to byteswap for multi-byte integers (16, 32, 64 bit)
                        if self.need_byteswap and type_code not in ('i', 'I'):
                            data = data.byteswap()
                        self._buffer.append(data.tobytes())
                elif np.issubdtype(dtype, np.bool_):
                    # Handle boolean type
                    if type_code == 'b':
                        # boolean
                        self._buffer.append(np.asarray(value, dtype=np.bool_).tobytes())
                elif np.issubdtype(dtype, np.floating):
                    # Handle floating point types
                    if type_code in ('h', 'f', 'd'):
                        # Map type codes to numpy dtypes
                        dtype_map = {'h': np.float16, 'f': np.float32, 'd': np.float64}
                        data = np.asarray(value, dtype=dtype_map[type_code])
                        if self.need_byteswap:
                            data = data.byteswap()
                        self._buffer.append(data.tobytes())
            else:
                # Default fallback for unsupported NumPy scalar types: convert to Python scalar
                self._write_element(value.item())
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def _write_numpy_array(self, arr: np.ndarray):
        """
        Write a NumPy array to the file.

        Supports various data types including integers, floats, booleans, and strings.
        Special handling is provided for string arrays, where an additional dimension
        is added to represent the string length for multi-dimensional string arrays.

        The array is automatically converted to C-contiguous order if needed,
        and byte swapping is performed if the system's endianness differs from the file's.

        Args:
            arr: The NumPy array to write with any supported dtype

        Raises:
            TypeError: If the array has an unsupported dtype
        """
        # Handle array shape
        for dim in arr.shape:
            self._write_length(dim)

        # Get the type code for the array's data type
        dtype = arr.dtype

        # Special handling for string arrays
        if np.issubdtype(dtype, np.string_) or np.issubdtype(dtype, np.unicode_):
            # For string arrays, we need to also write the string length dimension
            # Extract the itemsize which represents the max string length
            str_length = dtype.itemsize
            self._write_length(str_length)

            # For string arrays, use 's' type code
            self._buffer.append(b's')

            # Ensure the array is in C-contiguous order for efficient serialization
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)

            # Write the entire array memory to the file
            self._buffer.append(arr.tobytes())

            return

        if dtype not in self.type_map:
            raise TypeError(f"Unsupported NumPy dtype: {dtype}")

        type_code = self.type_map[dtype]
        self._buffer.append(type_code.encode())

        # Ensure the array is in C-contiguous order for efficient serialization
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        if self.need_byteswap:
            tobytes = lambda x: x.byteswap().tobytes()
        else:
            tobytes = lambda x: x.tobytes()

        # Write the array data based on its type
        if dtype == np.dtype('bool'):
            # Convert boolean array to bytes (0x00 for False, 0xFF for True)
            self._buffer.append(np.where(arr, 0xFF, 0x00).astype(np.uint8).tobytes())
        elif np.issubdtype(dtype, np.integer):
            # Handle integer types
            if type_code in ('i', 'I'):  # uint8, int8
                self._buffer.append(arr.tobytes())
            elif type_code in ('j', 'J'):  # uint16, int16
                self._buffer.append(tobytes(arr.astype(dtype)))
            elif type_code in ('k', 'K'):  # uint32, int32
                self._buffer.append(tobytes(arr.astype(dtype)))
            elif type_code in ('l', 'L'):  # uint64, int64
                self._buffer.append(tobytes(arr.astype(dtype)))
        elif np.issubdtype(dtype, np.floating):
            # Handle floating point types
            if type_code == 'h':  # float16
                self._buffer.append(tobytes(arr.astype(np.float16)))
            elif type_code == 'f':  # float32
                self._buffer.append(tobytes(arr.astype(np.float32)))
            elif type_code == 'd':  # float64
                self._buffer.append(tobytes(arr.astype(np.float64)))

    def _select_int_type(self, value: int) -> str:
        """
        Select the appropriate integer type code based on the value.

        Args:
            value: The integer value

        Returns:
            The xtype type code
        """
        if value >= 0:
            if value <= 0xFF:
                return 'I'  # uint8
            elif value <= 0xFFFF:
                return 'J'  # uint16
            elif value <= 0xFFFFFFFF:
                return 'K'  # uint32
            else:
                return 'L'  # uint64
        else:
            if value >= -0x80 and value <= 0x7F:
                return 'i'  # int8
            elif value >= -0x8000 and value <= 0x7FFF:
                return 'j'  # int16
            elif value >= -0x80000000 and value <= 0x7FFFFFFF:
                return 'k'  # int32
            else:
                return 'l'  # int64

    def _write_int_value(self, value: int, type_code: str):
        """
        Write an integer value with the specified type code.

        Args:
            value: The integer value
            type_code: The xtype type code
        """
        type_format = {
            'I': 'B', 'J': 'H', 'K': 'I', 'L': 'Q',  # unsigned
            'i': 'b', 'j': 'h', 'k': 'i', 'l': 'q'   # signed
        }
        format_char = type_format.get(type_code)
        if format_char:
            self._buffer.append(struct.pack(f'{self.struct_byteorder}{format_char}', value))

    def _write_length(self, length: int):
        """
        Write a length value using the appropriate format.

        Args:
            length: The length to write
        """
        if length <= 9:
            # Single-digit lengths are written as ASCII characters '0' through '9'
            self._buffer.append(str(length).encode())
        elif length <= 0xFF:
            # uint8 length
            self._buffer.append(b'M')
            self._buffer.append(struct.pack(f'{self.struct_byteorder}B', length))
        elif length <= 0xFFFF:
            # uint16 length
            self._buffer.append(b'N')
            self._buffer.append(struct.pack(f'{self.struct_byteorder}H', length))
        elif length <= 0xFFFFFFFF:
            # uint32 length
            self._buffer.append(b'O')
            self._buffer.append(struct.pack(f'{self.struct_byteorder}I', length))
        else:
            # uint64 length
            self._buffer.append(b'P')
            self._buffer.append(struct.pack(f'{self.struct_byteorder}Q', length))


class XTypeFileReader:
    """
    A class for reading Python data structures from files using the xtype binary format.
    """

    # Size in bytes for each xtype type code
    # This mapping is used to calculate memory requirements and array offsets
    type_sizes = {
        # Signed integers (1, 2, 4, 8 bytes)
        'i': 1, 'j': 2, 'k': 4, 'l': 8,
        # Unsigned integers (1, 2, 4, 8 bytes)
        'I': 1, 'J': 2, 'K': 4, 'L': 8,
        # Boolean (1 byte)
        'b': 1,
        # Floating point (2, 4, 8 bytes for float16, float32, float64)
        'h': 2, 'f': 4, 'd': 8,
        # String encodings
        's': 1,  # UTF-8 string (variable bytes per character)
        'u': 2,  # UTF-16 string (2 bytes per character)
        # Other types
        'S': 1,  # Struct type as array of bytes
        'x': 1,  # Generic byte array
    }

    # Map xtype type codes to NumPy dtypes
    # Used during deserialization to convert binary data to appropriate NumPy types
    dtype_map = {
        'b': np.bool_,     # Boolean
        'i': np.int8,      # 8-bit signed integer
        'j': np.int16,     # 16-bit signed integer
        'k': np.int32,     # 32-bit signed integer
        'l': np.int64,     # 64-bit signed integer
        'I': np.uint8,     # 8-bit unsigned integer
        'x': np.uint8,     # Raw bytes are treated as bytes or 8-bit unsigned integers
        'J': np.uint16,    # 16-bit unsigned integer
        'K': np.uint32,    # 32-bit unsigned integer
        'L': np.uint64,    # 64-bit unsigned integer
        'h': np.float16,   # 16-bit half-precision float
        'f': np.float32,   # 32-bit single-precision float
        'd': np.float64    # 64-bit double-precision float
    }

    def __init__(self, xtFile: File, byteorder: str = 'auto'):
        """
        Initialize an XTypeFileReader object.

        Args:
            file: The file object to read from
        """
        self.xtFile = xtFile
        self.file = xtFile.file
        self._pending_binary_size = 0
        self._pending_binary_type = None
        self.need_byteswap = False

        if byteorder == 'auto':
            # Read BOM to detect byte order automatically
            self._read_bom()
            if self.need_byteswap:
                self.byteorder = {'little': 'big', 'big': 'little'}[sys.byteorder]
            else:
                self.byteorder = sys.byteorder
        else:
            self.byteorder = byteorder
        self.struct_byteorder = {'little': '<', 'big': '>'}[self.byteorder]

    def _setPos(self, pos: int):
        """
        Set the file position to the given value.

        This method is useful for seeking to a specific position in the file
        and continuing to read from there.

        Args:
            pos: File position to seek to
        """
        self.file.seek(pos)
        self._pending_binary_size = 0

    def _getPos(self, withPendingBinary:bool = False) -> int:
        """
        Get the current file position.

        Returns the file position after any pending binary data.

        Returns:
            int: The current file position
        """
        pos = self.file.tell()
        if withPendingBinary:
            nRest = self._pending_binary_size
            if nRest:
                pos += nRest
        return pos

    def read(self, pos: int = 0) -> Any:
        """
        Read an xtype file and convert it to a Python object.

        This method is the counterpart to the write method. It reads the xtype file
        and returns a Python object that corresponds to what was written.

        Args:
            pos: File start position for reading

        Returns:
            Any: The Python object read from the file
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        # Reset the file position to the beginning
        self.file.seek(pos)
        self._pending_binary_size = 0

        # Start recursive parsing
        data = self._read_object()
        if type(data) is tuple:
            data = None
        return data

    def read_debug(self, indent_size: int = 2, max_indent_level: int = 10, max_binary_bytes: int = 15) -> Iterator[str]:
        """
        Iterator to read raw data from an xtype file and convert each output to a formatted string.

        This method uses read_raw and _read_raw_data to parse the xtype file and formats
        the output as a string where:
        - The string part is enclosed in quotation marks
        - If the string part ends with 's', the binary data is converted to a UTF-8 string with quotation marks
          unless it's part of a multidimensional array
        - Otherwise, the bytes are converted to hexadecimal with spaces in between
        - Indentation is added based on brackets [ and ] and braces { and }
        - Special bracket characters are always on their own line
        - String parts with empty binary data are gathered with a single pair of quotation marks
        - Length values are included in parentheses when present

        Args:
            indent_size: Number of spaces per indentation level (default: 2)
            max_indent_level: Maximum indentation level (default: 10)
            max_binary_bytes: Maximum number of binary bytes to read (default: 10)

        Yields:
            str: A formatted string for each element in the xtype file
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        # Reset the file position at the start
        self._setPos(0)

        # Initialize internal state
        indent_level = 0
        # Collecting characters that don't have binary data
        accumulated_strings = []
        # Track array shape to detect multidimensional arrays
        shape = []
        # Flag to indicate if we're inside an array context
        in_array_context = False

        try:
            for symbol, flag, length_or_size in self._read_raw():
                # Handle special bracket characters
                if symbol in '[]{}':
                    # First, if we have accumulated strings, yield them with the current indentation
                    if accumulated_strings:
                        # Join all accumulated strings without spaces and wrap in a single pair of quotes
                        accumulated_str = "".join(accumulated_strings)
                        yield ' ' * min(indent_level, max_indent_level) * indent_size + f'{accumulated_str}'
                        accumulated_strings = []

                    # For closing brackets, decrease indentation before printing
                    if symbol in ']}':
                        indent_level = max(0, indent_level - 1)
                        # If exiting an array context, reset shape
                        if symbol == ']' and in_array_context:
                            # Only reset if we're closing the outermost array
                            if indent_level == 0:
                                shape = []
                                in_array_context = False

                    # Print the bracket on its own line with proper indentation
                    yield ' ' * min(indent_level, max_indent_level) * indent_size + f'{symbol}'

                    # For opening brackets, increase indentation after printing
                    if symbol in '[{':
                        indent_level += 1
                        # Mark that we're entering an array context if it's a square bracket
                        if symbol == '[':
                            in_array_context = True

                    continue

                # For non-bracket characters with no binary data, accumulate them
                if flag == 0:
                    accumulated_strings.append(symbol)
                    continue
                elif flag == 1:
                    # If it's a length value (0-9 or M,N,O,P with value), include it in parentheses
                    accumulated_strings.append(f"{symbol}({length_or_size})" if symbol in "MNOP" else f"{symbol}")
                    # If we're in array context and get a length value, it could be a dimension
                    if in_array_context and indent_level > 0:
                        shape.append(length_or_size)
                    continue

                # If we get here, it's a data type with binary data (flag == 2)
                current_indent = ' ' * min(indent_level, max_indent_level) * indent_size

                # Include accumulated strings if any
                if accumulated_strings:
                    # Join all accumulated strings without spaces and add the current symbol
                    accumulated_str = "".join(accumulated_strings) + symbol
                    accumulated_strings = []
                else:
                    accumulated_str = symbol

                # Format based on accumulated_str
                if accumulated_str.endswith('s'):
                    # Get the string data and format them
                    binary_part = self._read_raw_data(max_bytes=max_binary_bytes) if length_or_size > 0 else b''
                    # Check if we're in a multidimensional array context
                    is_multidimensional = in_array_context and len(shape) > 1

                    # For string type, convert binary to UTF-8 string with quotation marks
                    # only if not part of a multidimensional array
                    try:
                        string_value = binary_part.decode('utf-8', errors='replace')
                        if is_multidimensional:
                            # Treat like any other non-string array - show as hex
                            hex_str = ' '.join(f'{b:02x}' for b in binary_part)
                            if len(binary_part) < length_or_size:
                                hex_str += f" ... ({length_or_size} bytes total)"
                            yield current_indent + f'{accumulated_str}: {hex_str}'
                        else:
                            # Regular string display with quotation marks
                            yield current_indent + f'{accumulated_str}: "{string_value}"'
                    except Exception:
                        # If decoding fails, fall back to hex representation
                        hex_str = ' '.join(f'{b:02x}' for b in binary_part)
                        if len(binary_part) < length_or_size:
                            hex_str += f" ... ({length_or_size} bytes total)"
                        yield current_indent + f'{accumulated_str}: {hex_str}'
                else:
                    # Get the data (limited by max_binary_bytes) and format them
                    binary_part = self._read_raw_data(max_bytes=max_binary_bytes) if length_or_size > 0 else b''
                    # For other types, convert to space-separated hex
                    hex_str = ' '.join(f'{b:02x}' for b in binary_part)
                    if len(binary_part) < length_or_size:
                        hex_str += f" ... ({length_or_size} bytes total)"
                    yield current_indent + f'{accumulated_str}: {hex_str}'
        except Exception as e:
            # If we have any accumulated strings when an exception occurs, output them
            if accumulated_strings:
                accumulated_str = "".join(accumulated_strings)
                yield ' ' * min(indent_level, max_indent_level) * indent_size + f'{accumulated_str}'
            # Get the current file position for debugging
            current_pos = self.file.tell()
            raise Exception(f"Error at file position {current_pos}: {str(e)}")
            # Don't re-raise the exception to allow partial output

    def _read_raw(self) -> Iterator[Tuple[str, int, int]]:
        """
        Iterator to read type information from an xtype file without consuming binary data.

        This method parses the file according to the xtype grammar and yields a tuple with:
        1. A string representing the symbol or type
        2. An integer flag indicating:
           - 0: No length or size information
           - 1: Length information
           - 2: Data size information
        3. The length or data size (0 if there's no length or size)

        If binary data is associated with the yielded type, the caller must call
        _read_raw_data() to consume the data or the subsequent _read_raw() call
        will skip over it automatically.

        Yields:
            Tuple[str, int, int]: (symbol/type, flag, length_or_size)
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        # Track accumulated length multipliers for arrays
        length_multiplier = 1

        while True:
            # Skip any pending binary data from previous call if not consumed
            if self._pending_binary_size > 0:
                self.file.seek(self._pending_binary_size, 1)  # Seek relative to current position
                self._pending_binary_size = 0

            # Read one byte
            char_byte = self.file.read(1)

            # Check for EOF
            if not char_byte:
                break

            try:
                char = char_byte.decode('ascii')
            except UnicodeDecodeError:
                # If we can't decode as ASCII, it's likely binary data that wasn't properly skipped
                # This can happen with string arrays where the binary data contains non-ASCII characters
                raise ValueError(f"Encountered non-ASCII character in grammar. This may indicate binary data wasn't properly skipped.")

            # Handle grammar terminal symbols
            if char in '[]{}TFn*':
                yield (char, 0, 0)
                continue

            # Handle direct length information (0-9)
            if char in '0123456789':
                yield (char, 1, int(char))
                # Multiply this length multiplier
                length_multiplier *= int(char)
                continue

            # Handle length information (M, N, O, P)
            if char in 'MNOP':
                size = {'M': 1, 'N': 2, 'O': 4, 'P': 8}[char]
                # binary_position = self.file.tell()
                binary_data = self.file.read(size)

                if len(binary_data) < size:
                    raise ValueError(f"Unexpected end of file when reading length of type {char}")

                # Convert binary to integer value based on type
                if char == 'M':  # uint8
                    value = binary_data[0]
                elif char == 'N':  # uint16
                    value = int.from_bytes(binary_data, byteorder=self.byteorder, signed=False)
                elif char == 'O':  # uint32
                    value = int.from_bytes(binary_data, byteorder=self.byteorder, signed=False)
                elif char == 'P':  # uint64
                    value = int.from_bytes(binary_data, byteorder=self.byteorder, signed=False)

                # Set pending binary size to 0 since we already consumed the binary data
                self._pending_binary_size = 0

                # Yield the length information and size
                yield (char, 1, value)

                # Multiply to length multiplier
                length_multiplier *= value
                continue

            # Handle data types
            if char in self.type_sizes:
                # For actual data types, calculate the total data size
                type_size = self.type_sizes[char]

                # Calculate total size based on accumulated length multiplier
                total_size = type_size * length_multiplier

                # Don't read the binary data yet, just note its size
                self._pending_binary_size = int(total_size)
                self._pending_binary_type = char

                yield (char, 2, total_size)
                length_multiplier = 1  # Reset length multiplier after using it
                continue

            # If we get here, we encountered an unexpected character
            raise ValueError(f"Unexpected character in xtype file: {repr(char)}")

    def _read_header(self) -> Tuple[str, int, List[int], List[Tuple]]:
        """
        Read headers from the file, collecting footnotes until a non-footnote is found.

        Returns:
            Tuple[str, int, List[int], List[Tuple]]: A tuple containing:
                - symbol: String representing the type or grammar symbol of the non-footnote element
                - size: Total size of the binary data in bytes
                - shape: List of shape for the non-footnote element
                - footnotes: List of all footnote elements (can be empty)
        """
        footnotes = []
        isFootnote = True

        while isFootnote:
            # Read the next header from the file
            symbol, size, shape = self._read_type()

            if symbol == '*':
                # This is a footnote marker
                # Read the footnote content that follows the marker
                footnote = ObjectProxy(self.xtFile, onlyContent=True)
                footnotes.append(footnote)
            else:
                # This is not a footnote, so we're done
                isFootnote = False

        return symbol, size, shape, footnotes

    def _read_type(self) -> Tuple[str, int, List[int]]:
        """
        Read a symbol and size information of an element from the xtype file.

        This method reads the xtype file using read_raw and returns the symbol and size information.
        The size information is the total size of the binary data associated with the symbol and
        the shape in case of arrays. If the symbol is a grammar symbol without binary data,
        the size is 0.

        Returns:
            Tuple[str, List[int], int]: A tuple containing:
                - symbol: String representing the type or grammar symbol
                - size: Total size of the binary data in bytes (0 for grammar
                        symbols without binary data)
                - shape: List of shape (empty for scalar values)
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        # Store length values (shape) for array types
        shape = []

        # Process raw elements until we find a complete logical element
        for symbol, flag, length_or_size in self._read_raw():
            # Case 1: Grammar terminals (single symbols)
            if symbol in '[]{}TFn*':
                return symbol, 0, []

            # Case 2: Length information (0-9, M, N, O, P)
            elif flag == 1:
                # Store dimension for array types
                shape.append(length_or_size)
                continue

            # Case 3: Data types with binary data
            elif flag == 2:
                # Return the data type with collected shape and size
                return symbol, length_or_size, shape

        # If we reach here, we've reached the end of the file
        return '', 0, []

    def _read_object(self) -> Any:
        """
        Read an object from the file.

        Returns:
            The Python object read from the file
        """

        symbol, size, shape, footnotes = self._read_header()

        if symbol in ('', ']', '}'):
            return (symbol,)
        else:
            return self._read_element(symbol, size, shape)

    def _read_element(self, symbol: str, size: int, shape: List[int]) -> Any:
        """
        Read an element based on its symbol from the file.

        Args:
            symbol: The symbol or type code read from the file
            size: The size of binary data in bytes (0 for grammar symbols)
            shape: List of shape for array types (empty for scalar values)

        Returns:
            The Python object read from the file
        """
        # Handle special symbols first
        if symbol == '[':
            # List
            return self._read_list()
        elif symbol == '{':
            # Dictionary
            return self._read_dict()
        elif symbol == 'T':
            # True
            return True
        elif symbol == 'F':
            # False
            return False
        elif symbol == 'n':
            # None
            return None
        elif symbol in self.type_sizes:
            # Check if this is an array type or a single element
            if shape:
                # This is an array type
                return self._read_numpy_array(shape, symbol, size)
            else:
                # This is a basic element (scalar, string or binary sequence)
                return self._read_basic_element(symbol, size)
        else:
            # Unexpected symbol
            raise ValueError(f"Unexpected symbol in xtype file: {symbol}")

    def _read_basic_element(self, type_code: str, size: int) -> Any:
        """
        Read a basic element from the file (scalar or string).

        Args:
            type_code: The xtype type code
            size: The total size of binary data in bytes

        Returns:
            The element read from the file
        """
        # Read the binary data
        binary_data = self._read_raw_data(size)

        # Parse based on type code
        if type_code == 'b':
            # Boolean
            return binary_data[0] != 0
        elif type_code in 'ijkl':
            # Signed integers
            return int.from_bytes(binary_data, byteorder=self.byteorder, signed=True)
        elif type_code in 'IJKL':
            # Unsigned integers
            return int.from_bytes(binary_data, byteorder=self.byteorder, signed=False)
        elif type_code in 'hfd':
            # Floating point
            if type_code == 'h':
                # float16
                return struct.unpack(f'{self.struct_byteorder}e', binary_data)[0]
            elif type_code == 'f':
                # float32
                return struct.unpack(f'{self.struct_byteorder}f', binary_data)[0]
            elif type_code == 'd':
                # float64
                return struct.unpack(f'{self.struct_byteorder}d', binary_data)[0]
        elif type_code == 's':
            # String
            return binary_data.decode('utf-8')
        elif type_code == 'x':
            # Bytes
            return binary_data
        else:
            # Unsupported type
            raise ValueError(f"Unsupported type code: {type_code}")

    def _read_list(self) -> List:
        """
        Read a list from the file.

        Returns:
            List: The list read from the file
        """
        result = []

        # Parse each element until we hit a closing bracket
        while True:
            symbol, size, shape = self._read_type()

            if symbol == ']' or symbol == '':
                # End of list
                break
            elif symbol in self.type_sizes:
                # Data type
                if shape:
                    # Array type
                    result.append(self._read_numpy_array(shape, symbol, size))
                else:
                    # Basic element
                    result.append(self._read_basic_element(symbol, size))
            else:
                # Special symbol or container
                result.append(self._read_element(symbol, size, shape))

        return result

    def _read_dict(self) -> Dict:
        """
        Read a dictionary from the file.

        Returns:
            Dict: The dictionary read from the file
        """
        result = {}

        # Parse key-value pairs until we hit a closing brace
        while True:
            # Read the key
            symbol, size, shape = self._read_type()

            if symbol == '}' or symbol == '':
                # End of dictionary
                break

            # We're reading a key, which should be a string
            if symbol == 's':
                # String key
                key_binary = self._read_raw_data(size)
                key = key_binary.decode('utf-8')
            elif symbol == 'u':
                # String key
                key_binary = self._read_raw_data(size)
                key = key_binary.decode('utf-16')
            elif symbol in 'ijklIJKL':
                if shape:
                    # Int array type
                    intArray = self._read_numpy_array(shape, symbol, size)
                    key = self._convert_to_deep_tuple(intArray.tolist())
                else:
                    # Int element
                    key = int(self._read_basic_element(symbol, size))
            elif symbol in 'hfd':
                if shape:
                    # Float array type
                    intArray = self._read_numpy_array(shape, symbol, size)
                    key = self._convert_to_deep_tuple(intArray.tolist())
                else:
                    # Float element
                    key = float(self._read_basic_element(symbol, size))
            else:
                # Unexpected symbol for key
                raise ValueError(f"Unexpected key type in dictionary: {symbol}")

            # Read the value
            symbol, size, shape = self._read_type()

            # We're reading a value
            if symbol in self.type_sizes:
                # Data type
                if shape and (symbol not in 'sx' or len(shape) > 1):
                    # Array type
                    result[key] = self._read_numpy_array(shape, symbol, size)
                else:
                    # Basic element
                    result[key] = self._read_basic_element(symbol, size)
            else:
                # Special symbol or container
                result[key] = self._read_element(symbol, size, shape)

        return result

    def _read_numpy_array(self, shape: List[int], type_code: str, size: int) -> np.ndarray:
        """
        Read a NumPy array from the file.

        Supports various data types including integers, floats, booleans, and strings.
        Special handling is provided for string arrays, where the last dimension
        represents the string length for multi-dimensional string arrays.

        Args:
            shape: The shape of the array (shape)
            type_code: The xtype type code
            size: The total size of binary data in bytes

        Returns:
            np.ndarray: The NumPy array read from the file with proper shape and data type
            str: For 1D string arrays, returns a Python string instead of an array

        Raises:
            ValueError: If an unsupported array type is encountered
        """
        # Read the binary data
        binary_data = self._read_raw_data(size)

        # Special handling for string arrays
        if type_code in 'sxu':
            # For 1D arrays, return a Python string
            if len(shape) == 1:
                if type_code == 's':
                    binary_data = binary_data.decode('utf-8')
                elif type_code == 'u':
                    binary_data = binary_data.decode('utf-16')
                # Decode the binary data as UTF-8 and return as a string
                return binary_data
            else:
                # For multidimensional arrays, the last dimension is the string length
                string_length = shape[-1]
                array_dims = shape[:-1]

                if type_code == 'u':
                    binary_data = binary_data.decode('utf-16').encode('utf-8')

                # Calculate total number of strings
                total_strings = 1
                for dim in array_dims:
                    total_strings *= dim

                # Create a numpy array of fixed-length strings
                string_array = np.empty(array_dims, dtype=f'S{string_length}')

                # Fill the array with the strings from binary_data
                flat_array = string_array.reshape(-1)

                for i in range(total_strings):
                    start = i * string_length
                    end = start + string_length
                    if start < len(binary_data):
                        # Get the string data, ensuring we don't go past the end of binary_data
                        string_data = binary_data[start:min(end, len(binary_data))]
                        # Pad with zeros if needed
                        if len(string_data) < string_length:
                            string_data = string_data.ljust(string_length, b'\x00')
                        flat_array[i] = string_data

                return string_array.reshape(array_dims)

        # Get the NumPy dtype
        if type_code not in self.dtype_map:
            raise ValueError(f"Unsupported NumPy type: {type_code}")

        dtype = self.dtype_map[type_code]

        # Calculate total number of elements
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        # Create a flat array first
        if type_code == 'b':
            # Handle boolean arrays specially (0x00 for False, anything else for True)
            flat_array = np.frombuffer(binary_data, dtype=np.uint8)
            flat_array = flat_array != 0
        elif type_code in 'jklJKLhfd':
            # Signed integers
            flat_array = np.frombuffer(binary_data, dtype=dtype)
        elif type_code in 'iIx':
            # Floating point
            flat_array = np.frombuffer(binary_data, dtype=dtype)
        else:
            # Unsupported type
            raise ValueError(f"Unsupported NumPy type: {type_code}")

        if self.need_byteswap:
            flat_array = flat_array.byteswap()

        # Reshape the array to the specified shape
        return flat_array.reshape(shape)

    def _convert_to_deep_tuple(self, lst: List) -> Tuple:
        """
        Convert a list to a deep tuple.

        Args:
            lst: The list to convert

        Returns:
            Tuple: The deep tuple
        """
        if not isinstance(lst, list):
            return lst
        return tuple(self._convert_to_deep_tuple(i) for i in lst)

    def _read_raw_data(self, max_bytes: int = None) -> bytes:
        """
        Read the binary data that corresponds to the last type yielded by _read_raw().

        This method must be called after _read_raw() yields a type with associated
        binary data. If not called, the next _read_raw() call will skip over the
        binary data automatically.

        Args:
            max_bytes: Maximum number of bytes to read. If None or greater than
                      remaining bytes, all remaining bytes are read. If less than
                      the total pending bytes, subsequent calls to _read_raw_data
                      can read the remaining bytes, or they will be skipped by
                      the next _read_raw() call.

        Returns:
            bytes: The binary data corresponding to the last type (up to max_bytes)

        Raises:
            ValueError: If there is no pending binary data to read. Call _read_raw() first.
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self._pending_binary_size == 0:
            return b''

        # Determine how many bytes to read
        bytes_to_read = self._pending_binary_size
        if max_bytes is not None and max_bytes < bytes_to_read:
            bytes_to_read = max_bytes

        # Read the binary data
        binary_data = self.file.read(bytes_to_read)
        if len(binary_data) < bytes_to_read:
            raise ValueError(f"Unexpected end of file when reading data of type {self._pending_binary_type}")

        # Update the pending binary size
        self._pending_binary_size -= bytes_to_read

        return binary_data

    def _read_bom(self):
        """
        Read the byte order mark (BOM) and adjust the byteorder if needed.

        The BOM is a 16-bit signed integer with the value 1234 written in big-endian byte order.
        If read with the wrong byte order, it would appear as -11772.

        This method checks for the BOM and switches the byteorder from big to little or vice versa
        if the BOM indicates a different byte order than the current setting.

        The file position is reset to the beginning after reading the BOM.
        """
        # Check if the current position is at the beginning of the file
        assert self.file.tell() == 0

        # Read the first two characters
        marker = self.file.read(2)

        # Check if the marker is '*J' which indicates a BOM follows
        if marker == b'*j':
            # Read the 2-byte integer using the current byteorder
            format_char = {'little': '<', 'big': '>'}[sys.byteorder]
            bom_value = struct.unpack(f'{format_char}h', self.file.read(2))[0]

            # If the value is -11772, we need to switch the byteorder
            self.need_byteswap = bom_value == -11772
        elif len(marker) == 0:
            raise EmptyFile
        else:
            # Reset the file position to the beginning
            self._setPos(0)
            self.need_byteswap = False
            return False

class ObjectProxy:
    """
    A class that represents an object of the xtype format with a pointer to a specific position in the file.

    This class allows for efficient navigation of xtype data structures without loading the entire object
    into memory, by tracking file positions and footnotes directly to the relevant parts of the file.
    """

    fileEnd, listEnd, dictEnd = [(i,) for i in ('',']','}')]

    def __init__(self, xtFile: File, position: int = -1, onlyContent: bool = False):
        """
        Initialize an ObjectProxy.

        Args:
            file: The xtype.File object
            position: The position in the file where the object starts
        """
        self.xtFile: File = xtFile
        self.reader: XTypeFileReader = xtFile.reader
        self.writer: XTypeFileWriter = xtFile.writer

        if position < 0:
            self.position = self.xtFile.file.tell()
        else:
            # Move file pointer to the specified position
            self.xtFile.file.seek(position)
            self.position = position

        if onlyContent:
            # Read only the object type
            symbol, size, shape = self.reader._read_type()
            footnotes = None
        else:
            # Read the object info (footnotes, type)
            symbol, size, shape, footnotes = self.reader._read_header()

        if not symbol:
            raise EOFError

        # Save data position
        self.data_position = self.reader._getPos()
        self.symbol = symbol
        self.data_size = size
        self.shape = shape
        self.footnotes = footnotes

    def _reset_reading(self) -> None:
        """Reset the reader position to the data position of this object."""
        # Move to the position of this object
        self.reader._setPos(self.data_position)

        # Set the size of binary object at position
        self.reader._pending_binary_size = self.data_size

    def __call__(self) -> Any:
        """
        Convert the entire object to a Python object.

        Returns:
            Any: The Python object read from the file
        """

        self.reader._setPos(self.data_position)
        self.reader._pending_binary_size = self.data_size

        if self.symbol in ('', ']', '}'):
            result = (self.symbol,)
        else:
            # Read the date and return the Python object
            result = self.reader._read_element(self.symbol, self.data_size, self.shape)
        return result

    def keys(self):
        """
        Return a list of keys from a dictionary object.

        Returns:
            list: List of keys from the dictionary

        Raises:
            TypeError: If the object is not a dictionary
        """

        assert self.reader is not None

        # Check if object is a dictionary
        if self.symbol != '{':
            raise TypeError(f"Object of type '{self.symbol}' is not a dictionary")

        self._reset_reading()

        # Read keys while skipping values
        keys = []
        while True:
            key_symbol, key_size, key_shape = self.reader._read_type()

            # Check if we've reached the end of the dictionary
            if key_symbol == '}':
                break

            # Read the key
            key = self.reader._read_element(key_symbol, key_size, key_shape)
            if isinstance(key, list):
                key = self.reader._convert_to_deep_tuple(key)

            keys.append(key)
            # Skip the value
            self._skip_object()

        return keys

    def __len__(self):
        """
        Return the length of a list, dictionary, or array object.

        For multi-dimensional arrays, returns the size of the first dimension.

        Returns:
            int: Number of items in the list, dictionary, or the first dimension of an array

        Raises:
            TypeError: If the object does not support length operations
        """

        assert self.reader is not None

        # Move to the reading position back
        self._reset_reading()

        # Check if object is a list or dictionary
        if self.symbol == '{':
            # For dictionaries, use the keys method to get the length
            return len(self.keys())
        elif self.symbol == '[':
            # For lists, count the number of items
            count = 0

            # Read items until end of list or EOF
            try:
                # Skip each item until we reach the end of the list (']')
                while True:
                    # Skip the item and check its returned symbol
                    symbol = self._skip_object()

                    # If we've reached the end of the list, stop counting
                    if symbol == ']':
                        break

                    # Otherwise, increment the counter
                    count += 1
            except EOFError:
                # Handle case where EOF closes the list
                pass

            self._reset_reading()
            return count

        # Handle arrays (non-container types with shape)
        elif self.shape and len(self.shape) > 0:

            # For arrays, return the first dimension size
            return self.shape[0]
        else:
            # Not a list, dictionary, or array
            raise TypeError(f"Object of type '{self.symbol}' does not support len()")

    def _get_item_value(self) -> Any:
        """
        Determine whether to return an ObjectProxy or a primitive value.

        This helper method reads the next object header and decides based on the type:
        - For containers (lists, dictionaries) and arrays: returns an ObjectProxy
        - For primitive types: reads and returns the actual object value directly

        Returns:
            Either an ObjectProxy instance or a primitive value depending on the object type
        """

        obj = ObjectProxy(self.xtFile)

        # Peek at the next object information to determine its type
        # symbol, size, shape, footnotes = object_header = self.reader._read_header()

        # Determine whether to return an ObjectProxy or the actual value
        if obj.symbol in '[{' or (obj.shape and (len(obj.shape) > 1 or obj.symbol not in 'sxu')):
            # Container type or array - return ObjectProxy
            return obj
        else:
            # Primitive type - read and return directly
            return obj()

    def _skip_object(self) -> str:
        """
        Skip over an object in the file.

        Handles footnotes and correctly counts opening and closing brackets/braces
        to ensure proper balance is maintained.

        Returns:
            str: The symbol that was found after skipping (usually the next element's symbol
             or a closing bracket/brace)
        """
        symbol = '-'
        nestedCount = 0
        while nestedCount > 0 or symbol == '-':
            next_symbol, size, shape = self.reader._read_type()
            if next_symbol == '*':
                # Skip over footnote content
                self._skip_object()
            elif next_symbol in '[{':
                nestedCount += 1
            elif next_symbol in ']}':
                nestedCount -= 1
            elif nestedCount == 0:
                symbol = next_symbol
                break
            else:
                symbol = next_symbol
            if symbol == '-':
                symbol = next_symbol
        return symbol

    def __getitem__(self, item: Union[int, str, slice, List[int], np.ndarray, Tuple]) -> Any:
        """
        Access a sub-element within the object using indexing operations.

        This method provides flexible indexing for different data structures:
        - For lists: Uses integer indices to access elements by position or slices for ranges
        - For dictionaries: Uses keys to access elements by key lookup
        - For arrays: Supports comprehensive array indexing including:
          - Integer indexing for single elements
          - Slice indexing for sub-arrays
          - List/array indexing for non-contiguous selections
          - Multi-dimensional indexing with tuples
          - Optimized block reading for contiguous slices

        Args:
            item: The index specifier, which can be:
                - Integer index (for lists and arrays)
                - Dictionary key (for dictionaries)
                - Slice object (for lists and arrays)
                - List/array of indices (for arrays)
                - Tuple of indices/slices/lists (for multi-dimensional arrays)

        Returns:
            For lists: An ObjectProxy pointing to the found object (for integer indices)
                      or a Python list (for slices)
            For dictionaries: An ObjectProxy pointing to the found object
            For arrays: The actual array data (NumPy array) or scalar value

        Raises:
            IndexError: If the index is out of bounds or the item is not found
            TypeError: If the object does not support the requested indexing operation
            ValueError: If an unsupported array type is encountered or invalid slice parameters
        """

        assert self.reader is not None

        # Move to the reading position back
        self._reset_reading()

        # Handle based on object type
        if self.symbol == '[':
            # Handle list indexing - both integer and slice access
            if isinstance(item, int):
                # Integer indexing - handle as before
                # Sequential access: read and skip objects until we reach the specified index
                # Note: This is O(n) access as we must traverse the list sequentially
                index = 0
                while index < item:
                    # Skip current object and check if we've reached the end of the list
                    next_symbol = self._skip_object()
                    if next_symbol == ']':
                        # We've reached the end of the list before finding the desired index
                        raise IndexError(f"List index {item} out of range, list has only {index} elements")
                    index += 1

                # Get the appropriate return value (ObjectProxy or primitive)
                return self._get_item_value()

            elif isinstance(item, slice):
                # Slice indexing - handle start, stop, step
                start = 0 if item.start is None else item.start
                stop = float('inf') if item.stop is None else item.stop
                step = 1 if item.step is None else item.step

                # Disallow negative start or stop values
                if start < 0:
                    raise ValueError("Negative start index is not supported for list slicing")
                if stop != float('inf') and stop < 0:
                    raise ValueError("Negative stop index is not supported for list slicing")

                # Validate step size
                if step < 0:
                    raise ValueError("Negative step size is not supported for list slicing")
                elif step == 0:
                    raise ValueError("Step size cannot be zero")

                # Create the list to store our results
                result = []
                next_symbol = '-'

                # Skip to start position
                index = 0
                while index < start:
                    next_symbol = self._skip_object()
                    index += 1

                # Read objects at positions that fall within the slice
                while index < stop:
                    # Check if we've reached the end of the list
                    if next_symbol == ']':
                        break

                    # Read the object directly without creating a new ObjectProxy
                    value = self.reader._read_object()
                    if type(value) is not tuple:
                        result.append(value)
                    elif value == self.listEnd and stop == float('inf'):
                        break
                    else:
                        raise IndexError(f"List index {index} out of range, unexpected symbol {value}")

                    # Skip (step-1) objects to get to the next desired position
                    for _ in range(step - 1):
                        # Skip and check if we've reached the end of the list
                        if self._skip_object() == ']':
                            if stop == float('inf'):
                                stop = index
                                break
                    index += step

                # Result is now complete
                return result

            else:
                raise TypeError(f"List indices must be integers or slices, got {type(item)}")

        elif self.symbol == '{':
            # Object is a dictionary - handle key-based lookup
            # Sequential scan through dictionary entries until we find the matching key
            while True:
                key_symbol, key_size, key_shape = self.reader._read_type()

                # Check if we've reached the end of the dictionary without finding the key
                if key_symbol == '}':
                    raise KeyError(f"Key {item} not found in dictionary")

                # Read the key and convert lists to tuples if needed (for hashability)
                key = self.reader._read_element(key_symbol, key_size, key_shape)
                if isinstance(key, list):
                    key = self.reader._convert_to_deep_tuple(key)

                if key == item:
                    # Key found, get the appropriate return value (ObjectProxy or primitive)
                    return self._get_item_value()
                else:
                    # Key doesn't match, skip the value and continue to next key-value pair
                    self._skip_object()

        elif self.shape and (len(self.shape) > 1 or self.symbol not in 'sxu'):
            # Get the current file position as the data start position
            data_start_pos = self.xtFile.file.tell()  # Position where the actual array data begins
            # Call the helper method for array handling to prepare variables
            dtype, index_arrays, result_shape, chunk_size, strides, element_size = \
                    self._handle_array_indexing(item)

            # Handle empty arrays
            if any(len(arr) == 0 for arr in index_arrays):
                return np.array([], dtype=dtype).reshape(result_shape)

            # Use itertools.product to iterate over all combinations of indices
            binary_data = []
            for indices in itertools.product(*index_arrays):
                # Calculate byte offset for this element using the original element size
                # The strides are based on dimension counts, not bytes
                offset = sum(idx * stride * element_size
                          for idx, stride in zip(indices, strides))

                # Seek to the position of this element and read the data
                self.xtFile.file.seek(data_start_pos + offset)
                element_bytes = self.xtFile.file.read(chunk_size)

                # Ensure we read the expected number of bytes - this could fail at EOF or with corrupted files
                assert len(element_bytes) == chunk_size
                binary_data.append(element_bytes)

            # Combine all binary data into a single buffer
            binary_buffer = b''.join(binary_data)

            # Create numpy array from binary data with the correct shape and dtype
            result = np.frombuffer(binary_buffer, dtype=dtype)

            # Reshape to match the shape of our result
            if result_shape:  # If we have shape, reshape; otherwise leave as 1D
                result = result.reshape(result_shape)

            # Correct the endianness if needed
            if self.reader.need_byteswap:
                result = result.byteswap()
            return result
        else:
            # Object is a singular type (int, float, str, etc.) which doesn't support indexing
            # This includes primitive types like integers, floats, strings, etc.
            raise TypeError(f"Object of type '{self.symbol}' does not support indexing")

    def __setitem__(self, item: Union[int, str, slice, Tuple], value: Any) -> None:
        """
        Assign a value to a sub-element within the object using indexing operations.

        This method provides array assignment operations, allowing for:
        - Single element assignment with integer indices
        - Sub-array assignment with slices
        - Assignment to non-contiguous selections with list/array indexing
        - Multi-dimensional indexing with tuples

        Args:
            item: The index specifier, which can be:
                - Integer index (for arrays)
                - Slice object (for arrays)
                - List/array of indices (for arrays)
                - Tuple of indices/slices/lists (for multi-dimensional arrays)
            value: The value to assign, must be compatible with the target's dtype and shape

        Raises:
            IndexError: If the index is out of bounds
            TypeError: If the object does not support assignment or index type is invalid
            ValueError: If shape or dtype mismatch between value and target
        """

        assert self.writer is not None

        # Move to the reading position back
        self._reset_reading()

        # Currently only array assignments are supported
        if not self.shape or (len(self.shape) <= 1 and self.symbol in 'sxu'):
            raise TypeError(f"Object of type '{self.symbol}' does not support item assignment")

        # Get the current file position as the data start position
        data_start_pos = self.xtFile.file.tell()  # Position where the actual array data begins

        # Call the helper method for array handling to prepare variables
        dtype, index_arrays, result_shape, chunk_size, strides, element_size = \
                self._handle_array_indexing(item)

        # Convert value to numpy array if it's not already one
        if not isinstance(value, np.ndarray):
            # If scalar value, convert to array with result_shape
            value = np.array(value, dtype=dtype)
            if result_shape and value.size == 1 and value.ndim == 0:
                # For scalar assignment to array subset, broadcast to the required shape
                value = np.full(result_shape, value.item(), dtype=dtype)

        # Check dtype compatibility
        if value.dtype != dtype:
            raise ValueError(f"Dtype mismatch: trying to assign {value.dtype} to array with dtype {dtype}")

        # Check shape compatibility
        if result_shape and value.shape != tuple(result_shape):
            raise ValueError(f"Shape mismatch: trying to assign array with shape {value.shape} to slice with shape {tuple(result_shape)}")

        # Make sure the file is in append mode to allow writing
        if None in (self.reader, self.writer):
            raise IOError("File must be opened in append mode ('a') for array assignment operations")

        # Use itertools.product to iterate over all combinations of indices
        # and assign the values accordingly
        flat_value = value.flatten() if hasattr(value, 'flatten') else np.array([value])

        # Check if we're writing a scalar value (either a single element array or a repeated value)
        is_scalar_assignment = (flat_value.size == 1 or np.isscalar(value))

        # Create a copy of flat_value and apply byteswap if needed
        if self.reader.need_byteswap:
            # Make a copy and apply numpy's byteswap method
            flat_value = flat_value.copy()
            flat_value = flat_value.byteswap()

        # For scalar values, prepare the byte sequence once
        scalar_bytes = None
        if is_scalar_assignment:
            scalar_bytes = flat_value[0].tobytes()

        flat_index = 0

        # Determine if we should use chunk-based writing
        elements_per_chunk = chunk_size // element_size
        use_chunks = elements_per_chunk > 1

        for indices in itertools.product(*index_arrays):
            # Calculate byte offset for this element using the original element size
            # The strides are based on dimension counts, not bytes
            offset = sum(idx * stride * element_size
                      for idx, stride in zip(indices, strides))

            # Seek to the position of this element
            self.xtFile.file.seek(data_start_pos + offset)

            # Handle writing based on chunk size
            if use_chunks:
                # Make sure we don't exceed the available values
                elements_to_write = min(elements_per_chunk, flat_value.size - flat_index)

                if is_scalar_assignment:
                    # Optimized scalar assignment - repeat the byte pattern to fill the chunk
                    binary_value = scalar_bytes * elements_per_chunk
                    # Trim to the correct length if needed (shouldn't exceed chunk_size)
                    binary_value = binary_value[:chunk_size]
                elif elements_to_write <= 0:
                    # Handle edge case where we're out of values but need to broadcast
                    if flat_value.size > 0:
                        # Broadcast the first value
                        chunk_values = np.full(elements_per_chunk, flat_value[0], dtype=dtype)
                        elements_to_write = elements_per_chunk
                    else:
                        raise ValueError("No values to assign")
                    # Write the chunk in binary form (byte swapping already applied if needed)
                    binary_value = chunk_values.tobytes()
                else:
                    # Get the chunk of values to write
                    chunk_values = flat_value[flat_index:flat_index + elements_to_write]
                    flat_index += elements_to_write
                    # Write the chunk in binary form (byte swapping already applied if needed)
                    binary_value = chunk_values.tobytes()
            else:
                # Single element writing (original approach)
                if is_scalar_assignment:
                    # For scalar assignments, we already have the bytes ready
                    binary_value = scalar_bytes
                elif flat_value.size > 0:
                    if flat_index < flat_value.size:
                        val_to_write = flat_value[flat_index]
                        flat_index += 1
                    else:
                        # In case we're broadcasting a single value
                        val_to_write = flat_value[0]
                    # Write the value in binary form (byte swapping already applied if needed)
                    binary_value = val_to_write.tobytes()
                else:
                    raise ValueError("No values to assign")

            # Write the data
            self.xtFile.file.write(binary_value)

    def __iter__(self):
        """
        Enable iteration over an ObjectProxy that points to a list.

        Returns:
            self: This instance as an iterator

        Raises:
            TypeError: If the object is not a list
        """

        # Store current position
        self._iter_original_pos = self.reader._getPos()

        # Move to the reading position back
        self._reset_reading()

        if self.symbol != '[':
            # Restore original position
            raise TypeError(f"Object of type '{self.symbol}' is not iterable")

        # Initialize iteration state
        self._iter_index = 0
        self._iter_done = False

        return self

    def __next__(self) -> Any:
        """
        Return the next item in the iteration.

        Returns:
            Any: The next item in the list

        Raises:
            StopIteration: When the end of the list is reached
        """
        if self._iter_done:
            # Reset file position to original position before raising StopIteration
            self.reader._setPos(self._iter_original_pos)
            raise StopIteration

        # Otherwise, read the object and increment index
        value = ObjectProxy(self.xtFile)()
        self._iter_index += 1

        # If we're at the end of the list, stop iteration
        if type(value) is tuple and value in (self.listEnd, self.fileEnd):
            self._iter_done = True
            self.reader._setPos(self._iter_original_pos)
            raise StopIteration

        return value

    def __repr__(self) -> str:
        """
        Return a string representation of the ObjectProxy.

        For arrays, includes shape information.
        For other types, shows the type symbol.

        Returns:
            str: A string representation of the ObjectProxy
        """
        # Type mapping for more readable output
        type_names = {
            'i': 'int8', 'j': 'int16', 'k': 'int32', 'l': 'int64',
            'I': 'uint8', 'J': 'uint16', 'K': 'uint32', 'L': 'uint64',
            'f': 'float32', 'd': 'float64',
            'b': 'bool',
            's': 'string', 'u': 'unicode', 'S': 'symbol',
            'x': 'bytes',
            '[': 'list', '{': 'dict',
            'T': 'true', 'F': 'false', 'n': 'null'
        }

        # Get a readable type name
        type_name = type_names.get(self.symbol, self.symbol)

        # For arrays (has shape and not a string/bytes type with only 1 dimension)
        if self.shape and (len(self.shape) > 1 or self.symbol not in 'sxu'):
            return f"<ObjectProxy type='{type_name}' shape={self.shape}>"
        else:
            return f"<ObjectProxy type='{type_name}'>"


    def _handle_array_indexing(self, item: Union[int, slice, List[int], np.ndarray, Tuple]) -> Tuple[np.dtype, List, List[int], int, List[int], int]:
        """
        Prepare variables for array indexing operations.

        This method handles the preparation logic for array indexing including:
        - Processing various index types (int, slice, list, numpy array)
        - Calculating strides and shapes for multi-dimensional arrays
        - Optimizing access patterns for different slice types
        - Memory-efficient handling of contiguous slices
        - Optimizing block reads for better performance

        Args:
            item: The index specifier (int, slice, list, numpy array, or tuple)

        Returns:
            Tuple containing:
                - dtype: NumPy data type for the array elements
                - index_arrays: List of index arrays for each dimension
                - result_shape: Shape of the resulting array
                - chunk_size: Size in bytes for each chunk to read
                - strides: List of stride values for each dimension
                - element_size: Size in bytes for each element

        Raises:
            IndexError: If the index is out of bounds
            TypeError: If the index type is invalid
            ValueError: If an unsupported array type is encountered
        """
        # Get element type information
        element_type = self.symbol  # The type code for array elements (i, f, d, etc.)
        element_size = self.reader.type_sizes[element_type]  # Size in bytes for each element

        # Map the xtype type code to the corresponding NumPy dtype
        if element_type not in self.reader.dtype_map:
            raise ValueError(f"Unsupported NumPy type: {element_type}")

        dtype = self.reader.dtype_map[element_type]

        # Normalize indexing to handle both single indices and tuples consistently
        if not isinstance(item, tuple):
            item_indices = (item,)  # Convert single index to a 1-tuple
        else:
            item_indices = item     # Use the tuple as is

        # Validate that we don't have more indices than shape
        if len(item_indices) > len(self.shape):
            raise IndexError(f"Too many indices for array with shape {self.shape}")

        # Calculate strides (elements to skip for each dimension)
        # Last dimension stride is 1 element
        strides = [1]
        for dim_size in reversed(self.shape[1:]):
            strides.insert(0, strides[0] * dim_size)

        # Process each index: convert integers to slices/tuples for iteration
        # This will store the final shape of our result
        result_shape = []
        # This will store the indices to access for each dimension
        index_arrays = []
        # This will store slice information (step, start, length) for each dimension
        slice_info = []

        # Process each dimension's index specification
        for i, (idx, dim_size) in enumerate(zip(item_indices, self.shape)):
            if isinstance(idx, int):
                # Single index: convert to a single-element tuple for iteration
                if idx < 0:
                    idx += dim_size  # Handle negative indexing
                if idx < 0 or idx >= dim_size:
                    raise IndexError(f"Index {idx} out of bounds for dimension {i} with size {dim_size}")
                index_arrays.append((idx,))  # No dimension in result shape (selecting single element)
                slice_info.append((0, 0, 0))  # Not a slice
            elif isinstance(idx, slice):
                # Slice: extract indices and create array
                start, stop, step = idx.indices(dim_size)
                indices = range(start, stop, step)
                index_arrays.append(indices)
                result_shape.append(len(indices))  # Add dimension to result shape
                slice_info.append((step, start, len(indices) if len(indices)!=dim_size else -1))  # Store slice parameters
            elif isinstance(idx, (list, np.ndarray)):
                # List or numpy array: use directly as indices
                indices = []
                for j in idx:
                    if isinstance(j, int):
                        if j < 0:
                            j += dim_size  # Handle negative indexing
                        if j < 0 or j >= dim_size:
                            raise IndexError(f"Index {j} out of bounds for dimension {i} with size {dim_size}")
                        indices.append(j)
                    else:
                        raise TypeError(f"Indices must be integers, not {type(j).__name__}")
                index_arrays.append(indices)
                result_shape.append(len(indices))  # Add dimension to result shape
            else:
                raise TypeError(f"Invalid index type: {type(idx).__name__}")

        chunk_size = element_size

        # Optimize by identifying full range slices with (1,0,-1) from the end
        full_range_count = 0
        for i in range(len(slice_info) - 1, -1, -1):
            if slice_info[i] == (1,0,-1):  # This represents a full slice with step=1, start=0, full length
                full_range_count += 1
            else:
                break

        # Reduce all 3 list variables by removing full range slices
        if full_range_count > 0:
            slice_info = slice_info[:-full_range_count]
            index_arrays = index_arrays[:-full_range_count]
            result_shape = result_shape[:-full_range_count]

        # Optimize contiguous memory access: if the last remaining slice has step 1 (contiguous memory)
        # we can read the whole slice at once instead of element by element
        if slice_info and slice_info[-1][0] == 1:  # step == 1 (contiguous memory)
            step, start, length = slice_info[-1]
            # Replace the last index_arrays element with just the start value
            # and increase chunk_size by the length factor to read all elements in one operation
            if length > 0:  # Make sure length is valid
                index_arrays[-1] = (start,)  # Just need the starting position
                chunk_size *= length  # Increase chunk size to read the entire slice at once

        # For any remaining dimensions not specified in item_indices (partial indexing case),
        # instead of creating full slices, increase the element_size to read data in larger chunks
        if len(index_arrays) < len(self.shape):
            # Calculate how much to increase element_size by multiplying by the sizes of all remaining dimensions
            remaining_dimensions_size = 1
            for i in range(len(index_arrays), len(self.shape)):
                remaining_dimensions_size *= self.shape[i]
                result_shape.append(self.shape[i])  # Add dimension to result shape

            # Increase chunk_size to read all data for remaining dimensions at once (for performance)
            chunk_size *= remaining_dimensions_size

        if len(index_arrays) == 0:
            index_arrays = [(0,)]

        # Return all the variables needed for the actual data reading
        return dtype, index_arrays, result_shape, chunk_size, strides, element_size
