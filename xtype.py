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

__version__ = "0.2.0"

import struct
import numpy as np
from typing import Any, Dict, List, Tuple, Union, BinaryIO, Iterator, Optional, Callable
import itertools

DEFAULT_BYTE_ORDER = 'big'

# Size in bytes for each xtype type code
# This mapping is used to calculate memory requirements and array offsets
TYPE_SIZES = {
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

# Type mapping between Python/NumPy types and xtype format type codes
# Used during serialization to determine the appropriate type code for each value
TYPE_MAP = {
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

# Map xtype type codes to NumPy dtypes
# Used during deserialization to convert binary data to appropriate NumPy types
DTYPE_MAP = {
    'b': np.bool_,     # Boolean
    'i': np.int8,      # 8-bit signed integer
    'j': np.int16,     # 16-bit signed integer
    'k': np.int32,     # 32-bit signed integer
    'l': np.int64,     # 64-bit signed integer
    'I': np.uint8,     # 8-bit unsigned integer
    'x': np.uint8,     # Raw bytes are treated as 8-bit unsigned integers
    'J': np.uint16,    # 16-bit unsigned integer
    'K': np.uint32,    # 32-bit unsigned integer
    'L': np.uint64,    # 64-bit unsigned integer
    'h': np.float16,   # 16-bit half-precision float
    'f': np.float32,   # 32-bit single-precision float
    'd': np.float64    # 64-bit double-precision float
}

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
# <type>       ::= <lenght> <type> | <bin_data>
# <lenght>     ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" |
#                  "M" <bin_data> | "N" <bin_data> | "O" <bin_data> | "P" <bin_data>
# <bin_type>   ::= "i" | "j" | "k" | "l" | "I" | "J" | "K" | "L" |
#                  "b" | "h" | "f" | "d" | "s" | "u" | "S" | "x"

# <bin_data> represents the actual binary data of the specified size for the given type.
# <EOF> marks the end of file. In streaming applications, this could be represented by a zero byte.


class File:
    """
    A class for reading and writing Python data structures to files using the xtype binary format.

    Supports serialization of:
    - Basic types: int, float, str, bytes, bool
    - Container types: list, dict
    - NumPy arrays: 1D, 2D, and higher-dimensional arrays with various data types
    """

    def __init__(self, filename: str, mode: str = 'r'):
        """
        Initialize an xtype.File object.

        Args:
            filename: Path to the file
            mode: File mode ('w' for write, 'r' for read)
        """
        self.filename = filename
        self.mode = mode
        self.file = None

        # Reader and writer instances (initialized in open())
        self.reader = None
        self.writer = None

    def __enter__(self):
        """Context manager entry point."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()

    def open(self):
        """Open the file for reading or writing."""
        if self.mode == 'w':
            self.file = open(self.filename, 'wb')
            self.writer = XTypeFileWriter(self.file)
        elif self.mode == 'r':
            self.file = open(self.filename, 'rb')
            self.reader = XTypeFileReader(self.file)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def close(self):
        """Close the file."""
        if self.file and not self.file.closed:
            self.file.close()

    def write(self, data: Any):
        """
        Write a Python object to the file in xtype format.

        Args:
            data: The Python object to serialize (can be a primitive type,
                 list, dict, or numpy array)
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for writing")

        if self.mode != 'w':
            raise IOError("File is not open in write mode")

        self.writer._write_bom()
        self.writer._write_object(data)

    def read(self, byteorder: str = 'auto') -> Any:
        """
        Read an xtype file and convert it to a Python object.

        Args:
            byteorder: The byte order of multi-byte integers in the file.
                       'big', 'little' or 'auto'. Defaults to 'auto'.

        Returns:
            Any: The Python object read from the file
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        # Set byte order
        if byteorder in ('little', 'big'):
            self.reader.byteorder = byteorder

        # Reset the file position to the beginning
        self.file.seek(0)

        # If byteorder is 'auto', try to detect it from the BOM
        if byteorder == 'auto':
            self.reader._read_bom()

        # Start recursive parsing
        return self.reader.read()

    def __getitem__(self, key):
        """
        Access an element within the file.

        Creates an objPointer at the beginning of the file and uses its __getitem__ method.

        Args:
            key: Index (for lists), key (for dicts), or slice (for arrays)

        Returns:
            objPointer: A new objPointer pointing to the beginning of the found object
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        # Create an objPointer at the beginning of the file
        root = objPointer(self, 0)

        # Use the objPointer's __getitem__ method
        return root[key]

    def read_debug(self, indent_size: int = 2, max_indent_level: int = 10, byteorder: str = 'auto', max_binary_bytes: int = 15) -> Iterator[str]:
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

        return self.reader.read_debug(indent_size, max_indent_level, byteorder, max_binary_bytes)

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

        # Create an objPointer at the beginning of the file
        ptr = objPointer(self, 0)
        return ptr.keys()

class XTypeFileWriter:
    """
    A class for writing Python data structures to files using the xtype binary format.
    """

    # Default byteorder is big-endian ('>') as used in all struct.pack calls
    byteorder = DEFAULT_BYTE_ORDER

    # Struct format character for byteorder
    _struct_byteorder_format = {
        'little': '<',
        'big': '>'
    }

    def __init__(self, file: BinaryIO, byteorder: str = 'auto'):
        """
        Initialize an XTypeFileWriter object.

        Args:
            file: The file object to write to
        """
        self.file = file

        if byteorder in ('little', 'big'):
            self.byteorder = byteorder

        # Type mapping between Python/NumPy types and xtype format types
        self.type_map = TYPE_MAP

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
        self.file.write(b'*j')
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
            self.file.write(b'n')
        else:
            self._write_element(obj)

    def _write_list(self, lst: List):
        """
        Write a list to the file.

        Args:
            lst: The list to write
        """
        self.file.write(b'[')
        for item in lst:
            self._write_object(item)
        self.file.write(b']')

    def _write_dict(self, d: Dict):
        """
        Write a dictionary to the file.

        Args:
            d: The dictionary to write
        """
        self.file.write(b'{')
        for key, value in d.items():
            # Convert key to string if it's not already
            if not isinstance(key, str):
                key = str(key)
            # Write the key as a string element
            self._write_element(key)
            # Write the value
            self._write_object(value)
        self.file.write(b'}')

    def _write_element(self, value: Any):
        """
        Write a basic element to the file.

        Args:
            value: The value to write
        """
        if value is None:
            self.file.write(b'n')
        elif isinstance(value, bool):
            self.file.write(b'T' if value else b'F')
        elif isinstance(value, int):
            type_code = self._select_int_type(value)
            self.file.write(type_code.encode())
            self._write_int_value(value, type_code)
        elif isinstance(value, float):
            self.file.write(b'd')
            struct_format = self._struct_byteorder_format.get(self.byteorder, '>')
            self.file.write(struct.pack(f'{struct_format}d', value))
        elif isinstance(value, str):
            # Write string with length prefix
            encoded = value.encode('utf-8')
            self._write_length(len(encoded))
            self.file.write(b's')
            self.file.write(encoded)
        elif isinstance(value, bytes):
            # Write bytes with length prefix
            self._write_length(len(value))
            self.file.write(b'x')
            self.file.write(value)
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def _write_numpy_array(self, arr: np.ndarray):
        """
        Write a NumPy array to the file.

        Args:
            arr: The NumPy array to write
        """
        # Handle array dimensions
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
            self.file.write(b's')

            # Ensure the array is in C-contiguous order for efficient serialization
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)

            # Write the entire array memory to the file
            self.file.write(arr.tobytes())

            return

        if dtype not in self.type_map:
            raise TypeError(f"Unsupported NumPy dtype: {dtype}")

        type_code = self.type_map[dtype]
        self.file.write(type_code.encode())

        # Ensure the array is in C-contiguous order for efficient serialization
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        # Write the array data based on its type
        if dtype == np.dtype('bool'):
            # Convert boolean array to bytes (0x00 for False, 0xFF for True)
            self.file.write(np.where(arr, 0xFF, 0x00).astype(np.uint8).tobytes())
        elif np.issubdtype(dtype, np.integer):
            # Handle integer types
            if type_code in ('i', 'I'):  # uint8, int8
                self.file.write(arr.tobytes())
            elif type_code in ('j', 'J'):  # uint16, int16
                self.file.write(arr.astype(dtype).byteswap(self.byteorder == 'big').tobytes())
            elif type_code in ('k', 'K'):  # uint32, int32
                self.file.write(arr.astype(dtype).byteswap(self.byteorder == 'big').tobytes())
            elif type_code in ('l', 'L'):  # uint64, int64
                self.file.write(arr.astype(dtype).byteswap(self.byteorder == 'big').tobytes())
        elif np.issubdtype(dtype, np.floating):
            # Handle floating point types
            if type_code == 'h':  # float16
                self.file.write(arr.astype(np.float16).byteswap(self.byteorder == 'big').tobytes())
            elif type_code == 'f':  # float32
                self.file.write(arr.astype(np.float32).byteswap(self.byteorder == 'big').tobytes())
            elif type_code == 'd':  # float64
                self.file.write(arr.astype(np.float64).byteswap(self.byteorder == 'big').tobytes())

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
        struct_format = self._struct_byteorder_format.get(self.byteorder, '>')
        if type_code == 'I':
            self.file.write(struct.pack(f'{struct_format}B', value))
        elif type_code == 'J':
            self.file.write(struct.pack(f'{struct_format}H', value))
        elif type_code == 'K':
            self.file.write(struct.pack(f'{struct_format}I', value))
        elif type_code == 'L':
            self.file.write(struct.pack(f'{struct_format}Q', value))
        elif type_code == 'i':
            self.file.write(struct.pack(f'{struct_format}b', value))
        elif type_code == 'j':
            self.file.write(struct.pack(f'{struct_format}h', value))
        elif type_code == 'k':
            self.file.write(struct.pack(f'{struct_format}i', value))
        elif type_code == 'l':
            self.file.write(struct.pack(f'{struct_format}q', value))

    def _write_length(self, length: int):
        """
        Write a length value using the appropriate format.

        Args:
            length: The length to write
        """
        struct_format = self._struct_byteorder_format.get(self.byteorder, '>')
        if length <= 9:
            # Single-digit lengths are written as ASCII characters '0' through '9'
            self.file.write(str(length).encode())
        elif length <= 0xFF:
            # uint8 length
            self.file.write(b'M')
            self.file.write(struct.pack(f'{struct_format}B', length))
        elif length <= 0xFFFF:
            # uint16 length
            self.file.write(b'N')
            self.file.write(struct.pack(f'{struct_format}H', length))
        elif length <= 0xFFFFFFFF:
            # uint32 length
            self.file.write(b'O')
            self.file.write(struct.pack(f'{struct_format}I', length))
        else:
            # uint64 length
            self.file.write(b'P')
            self.file.write(struct.pack(f'{struct_format}Q', length))


class XTypeFileReader:
    """
    A class for reading Python data structures from files using the xtype binary format.
    """

    # Default byteorder is big-endian as used in all read methods
    byteorder = DEFAULT_BYTE_ORDER

    # Struct format character for byteorder
    _struct_byteorder_format = {
        'little': '<',
        'big': '>'
    }

    def __init__(self, file: BinaryIO):
        """
        Initialize an XTypeFileReader object.

        Args:
            file: The file object to read from
        """
        self.file = file
        self._pending_binary_size = 0
        self._pending_binary_type = None

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

    def _getPos(self) -> int:
        """
        Get the current file position.

        Returns the file position after any pending binary data.

        Returns:
            int: The current file position
        """
        pos = self.file.tell()
        nRest = self._pending_binary_size
        if nRest is not None:
            pos += nRest
        return pos

    def read(self, byteorder: str = 'auto', pos: int = 0) -> Any:
        """
        Read an xtype file and convert it to a Python object.

        This method is the counterpart to the write method. It reads the xtype file
        and returns a Python object that corresponds to what was written.

        Args:
            byteorder: The byte order of multi-byte integers in the file.
                       'big', 'little' or 'auto'. Defaults to 'auto'.
            pos: File start position for reading

        Returns:
            Any: The Python object read from the file
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        # Set byte order
        if byteorder in ('little', 'big'):
            self.byteorder = byteorder

        # Reset the file position to the beginning
        self.file.seek(pos)
        self.file._pending_binary_size = 0

        # If byteorder is 'auto', try to detect it from the BOM
        if byteorder == 'auto':
            self._read_bom()

        # Start recursive parsing
        return self._read_object()

    def read_debug(self, indent_size: int = 2, max_indent_level: int = 10, byteorder: str = 'auto', max_binary_bytes: int = 15) -> Iterator[str]:
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
            byteorder: The byte order of multi-byte integers in the file.
                       'big', 'little' or 'auto'. Defaults to 'auto'.
            max_binary_bytes: Maximum number of binary bytes to read (default: 10)

        Yields:
            str: A formatted string for each element in the xtype file
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")


        # Set byte order
        if byteorder in ('little', 'big'):
            self.byteorder = byteorder

        # Reset the file position at the start
        self._setPos(0)

        # Initialize internal state
        indent_level = 0
        # Collecting characters that don't have binary data
        accumulated_strings = []
        # Track array dimensions to detect multidimensional arrays
        dimensions = []
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
                        # If exiting an array context, reset dimensions
                        if symbol == ']' and in_array_context:
                            # Only reset if we're closing the outermost array
                            if indent_level == 0:
                                dimensions = []
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
                        dimensions.append(length_or_size)
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
                    is_multidimensional = in_array_context and len(dimensions) > 1

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
            if char in TYPE_SIZES:
                # For actual data types, calculate the total data size
                type_size = TYPE_SIZES[char]

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

    def _read_header(self) -> Tuple[str, List[int], int]:
        """
        Read a symbol and size information of an element from the xtype file.

        This method reads the xtype file using read_raw and returns the symbol and size information.
        The size information is the total size of the binary data associated with the symbol and
        the dimensions in case of arrays. If the symbol is a grammar symbol without binary data,
        the size is 0.

        Returns:
            Tuple[str, List[int], int]: A tuple containing:
                - symbol: String representing the type or grammar symbol
                - size: Total size of the binary data in bytes (0 for grammar
                        symbols without binary data)
                - dimensions: List of dimensions (empty for scalar values)
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        # Store length values (dimensions) for array types
        dimensions = []

        # Process raw elements until we find a complete logical element
        for symbol, flag, length_or_size in self._read_raw():
            # Case 1: Grammar terminals (single symbols)
            if symbol in '[]{}TFn*':
                return symbol, 0, []

            # Case 2: Length information (0-9, M, N, O, P)
            elif flag == 1:
                # Store dimension for array types
                dimensions.append(length_or_size)
                continue

            # Case 3: Data types with binary data
            elif flag == 2:
                # Return the data type with collected dimensions and size
                return symbol, length_or_size, dimensions

        # If we reach here, we've reached the end of the file
        return '', 0, []

    def _read_object(self) -> Any:
        """
        Read an object from the file.

        Returns:
            The Python object read from the file
        """

        isFootnote = True
        while isFootnote:
            # Read the next step from the file
            symbol, size, dimensions = self._read_header()
            if symbol == '*':
                symbol, size, dimensions = self._read_header()
            else:
                isFootnote = False

        return self._read_element(symbol, size, dimensions)

    def _read_element(self, symbol: str, size: int, dimensions: List[int]) -> Any:
        """
        Read an element based on its symbol from the file.

        Args:
            symbol: The symbol or type code read from the file
            size: The size of binary data in bytes (0 for grammar symbols)
            dimensions: List of dimensions for array types (empty for scalar values)

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
        elif symbol in TYPE_SIZES:
            # Check if this is an array type or a single element
            if dimensions:
                # This is an array type
                return self._read_numpy_array(dimensions, symbol, size)
            else:
                # This is a basic element (scalar or string)
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
            struct_format = self._struct_byteorder_format.get(self.byteorder, '<')
            if type_code == 'h':
                # float16
                return struct.unpack(f'{struct_format}e', binary_data)[0]
            elif type_code == 'f':
                # float32
                return struct.unpack(f'{struct_format}f', binary_data)[0]
            elif type_code == 'd':
                # float64
                return struct.unpack(f'{struct_format}d', binary_data)[0]
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
            symbol, size, dimensions = self._read_header()

            if symbol == ']':
                # End of list
                break
            elif symbol in TYPE_SIZES:
                # Data type
                if dimensions:
                    # Array type
                    result.append(self._read_numpy_array(dimensions, symbol, size))
                else:
                    # Basic element
                    result.append(self._read_basic_element(symbol, size))
            else:
                # Special symbol or container
                result.append(self._read_element(symbol, size, dimensions))

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
            symbol, size, dimensions = self._read_header()

            if symbol == '}':
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
                if dimensions:
                    # Int array type
                    intArray = self._read_numpy_array(dimensions, symbol, size)
                    key = self._convert_to_deep_tuple(intArray.tolist())
                else:
                    # Int element
                    key = int(self._read_basic_element(symbol, size))
            elif symbol in 'hfd':
                if dimensions:
                    # Float array type
                    intArray = self._read_numpy_array(dimensions, symbol, size)
                    key = self._convert_to_deep_tuple(intArray.tolist())
                else:
                    # Float element
                    key = float(self._read_basic_element(symbol, size))
            else:
                # Unexpected symbol for key
                raise ValueError(f"Unexpected key type in dictionary: {symbol}")

            # Read the value
            symbol, size, dimensions = self._read_header()

            # We're reading a value
            if symbol in TYPE_SIZES:
                # Data type
                if dimensions and (symbol not in 'sx' or len(dimensions) > 1):
                    # Array type
                    result[key] = self._read_numpy_array(dimensions, symbol, size)
                else:
                    # Basic element
                    result[key] = self._read_basic_element(symbol, size)
            else:
                # Special symbol or container
                result[key] = self._read_element(symbol, size, dimensions)

        return result

    def _read_numpy_array(self, dimensions: List[int], type_code: str, size: int) -> np.ndarray:
        """
        Read a NumPy array from the file.

        Args:
            dimensions: The dimensions of the array
            type_code: The xtype type code
            size: The total size of binary data in bytes

        Returns:
            np.ndarray: The NumPy array read from the file
        """
        # Read the binary data
        binary_data = self._read_raw_data(size)

        # Special handling for string arrays
        if type_code == 's':
            # For 1D arrays, return a Python string
            if len(dimensions) == 1:
                # Decode the binary data as UTF-8 and return as a string
                return binary_data.decode('utf-8')
            else:
                # For multidimensional arrays, the last dimension is the string length
                string_length = dimensions[-1]
                array_dims = dimensions[:-1]

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
        if type_code not in DTYPE_MAP:
            raise ValueError(f"Unsupported NumPy type: {type_code}")

        dtype = DTYPE_MAP[type_code]

        # Calculate total number of elements
        total_elements = 1
        for dim in dimensions:
            total_elements *= dim

        # Create a flat array first
        if type_code == 'b':
            # Handle boolean arrays specially (0x00 for False, anything else for True)
            flat_array = np.frombuffer(binary_data, dtype=np.uint8)
            flat_array = flat_array != 0
        elif type_code in 'jklJKLhfd':
            # Signed integers (need to be byteswapped because xtype uses big-endian)
            flat_array = np.frombuffer(binary_data, dtype=dtype)
            flat_array = flat_array.byteswap()
        elif type_code in 'iIx':
            # Floating point (need to be byteswapped because xtype uses big-endian)
            flat_array = np.frombuffer(binary_data, dtype=dtype)
        else:
            # Unsupported type
            raise ValueError(f"Unsupported NumPy type: {type_code}")

        # Reshape the array to the specified dimensions
        return flat_array.reshape(dimensions)

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
            ValueError: If there is no pending binary data to read
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self._pending_binary_size <= 0:
            raise ValueError("No pending binary data to read. Call _read_raw() first.")

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
        # Save the current position
        current_pos = self.file.tell()

        # Read the first two characters
        marker = self.file.read(2)

        # Check if the marker is '*J' which indicates a BOM follows
        if marker == b'*j':
            # Read the 2-byte integer using the current byteorder
            format_char = self._struct_byteorder_format[self.byteorder]
            bom_value = struct.unpack(f'{format_char}h', self.file.read(2))[0]

            # If the value is -11772, we need to switch the byteorder
            if bom_value == -11772:
                self.byteorder = 'little' if self.byteorder == 'big' else 'big'
            elif bom_value != 1234:
                # If it's neither 1234 nor -11772, it's not a valid BOM
                pass  # Keep the default byteorder
        else:
            # Reset the file position to the beginning
            self._setPos(current_pos)


class objPointer:
    """
    A class that represents an object of the xtype format with a pointer to a specific position in the file.

    This class allows for efficient navigation of xtype data structures without loading the entire object
    into memory, by tracking file positions and navigating directly to the relevant parts of the file.
    """

    def __init__(self, file: File, position: int):
        """
        Initialize an objPointer.

        Args:
            file: The xtype.File object
            position: The position in the file where the object starts
        """
        self.file = file
        self.reader = file.reader
        self.writer = file.writer

        self.position = position
        self.footnotes = []

        # Move file pointer to the specified position
        self.file.file.seek(self.position)
        self.file._pending_binary_size = 0

    def _skip_object(self):
        """
        Skip over an object in the file.

        Handles footnotes and correctly counts opening and closing brackets/braces
        to ensure proper balance is maintained.
        """
        nestedCount = None
        while nestedCount != 0:
            if nestedCount is None:
                nestedCount = 0
            symbol, size, dimensions = self.reader._read_header()
            if symbol == '*':
                # Skip over footnote content
                self._skip_object()
            elif symbol in '[{':
                nestedCount += 1
            elif symbol in ']}':
                nestedCount -= 1
            elif nestedCount == 0:
                break

    def __getitem__(self, item):
        """
        Access a sub-element within the object using indexing operations.

        This method provides flexible indexing for different data structures:
        - For lists: Uses integer indices to access elements by position
        - For dictionaries: Uses keys to access elements by key lookup
        - For arrays: Supports both direct element access and sub-array slicing
          with multi-dimensional indexing capabilities

        Args:
            item: The index specifier, which can be:
                - Integer index (for lists and arrays)
                - Dictionary key (for dictionaries)
                - Tuple of indices (for multi-dimensional arrays)

        Returns:
            For lists and dictionaries: An objPointer pointing to the found object
            For arrays: The actual array data (NumPy array) or scalar value

        Raises:
            IndexError: If the index is out of bounds or the item is not found
            TypeError: If the object does not support the requested indexing operation
            ValueError: If an unsupported array type is encountered
        """
        # Store current position to restore it later if needed
        self.reader._setPos(self.position)

        # Re-read the object type and handle footnotes
        is_footnote = True
        while is_footnote:
            symbol, size, dimensions = self.reader._read_header()
            if symbol == '*':
                # Skip over footnote without reading its content
                self.reader._read_object()
            else:
                is_footnote = False

        # Handle based on object type
        if symbol == '[':
            # Object is a list - handle integer indexing
            if not isinstance(item, int):
                raise TypeError(f"List indices must be integers, got {type(item)}")

            # Sequential access: read and skip objects until we reach the specified index
            # Note: This is O(n) access as we must traverse the list sequentially
            index = 0
            while index < item:
                self._skip_object()
                index += 1

            # Create a new objPointer at the current position (pointing to the target element)
            return objPointer(self.file, self.reader._getPos())

        elif symbol == '{':
            # Object is a dictionary - handle key-based lookup
            # Sequential scan through dictionary entries until we find the matching key
            while True:
                key_symbol, key_size, key_dimensions = self.reader._read_header()

                # Check if we've reached the end of the dictionary without finding the key
                if key_symbol == '}':
                    raise KeyError(f"Key {item} not found in dictionary")

                # Read the key and convert lists to tuples if needed (for hashability)
                key = self.reader._read_element(key_symbol, key_size, key_dimensions)
                if isinstance(key, list):
                    key = self.reader._convert_to_deep_tuple(key)

                if key == item:
                    # Key found, return a new objPointer for the associated value
                    return objPointer(self.file, self.reader._getPos())
                else:
                    # Key doesn't match, skip the value and continue to next key-value pair
                    self._skip_object()

        elif dimensions:
            # Object is an array - handle direct binary reading with efficient random access
            # This provides O(1) access to array elements regardless of position

            # Get element type information
            element_type = symbol  # The type code for array elements (i, f, d, etc.)
            element_size = TYPE_SIZES[element_type]  # Size in bytes for each element
            data_start_pos = self.file.file.tell()  # Position where the actual array data begins

            # Map the xtype type code to the corresponding NumPy dtype
            if element_type not in DTYPE_MAP:
                raise ValueError(f"Unsupported NumPy type: {element_type}")

            dtype = DTYPE_MAP[element_type]

            # Normalize indexing to handle both single indices and tuples consistently
            if not isinstance(item, tuple):
                indices = (item,)  # Convert single index to a 1-tuple
            else:
                indices = item     # Use the tuple as is

            # Validate that we don't have more indices than dimensions
            if len(indices) > len(dimensions):
                raise IndexError(f"Too many indices for array with shape {dimensions}")

            # Determine if byte swapping is needed based on endianness differences
            # between the file's byte order and the system's native byte order
            need_byteswap = (self.reader.byteorder == 'big') != (np.dtype(dtype).byteorder == '>')

            # Handle array indexing based on dimensions and provided indices

            # Case 1: Complete indexing - accessing a single element (scalar value)
            # When the number of indices matches the number of dimensions, we're accessing a single value
            if len(indices) == len(dimensions):
                # Calculate memory offset for the target element using multi-dimensional indexing formula
                # offset = i₁ × (s₂ × s₃ × ... × sₙ) + i₂ × (s₃ × ... × sₙ) + ... + iₙ₋₁ × sₙ + iₙ
                # where i is the index and s is the dimension size
                offset = 0
                for i, idx in enumerate(indices):
                    # Calculate stride (element spacing) for this dimension
                    stride = element_size  # Start with basic element size
                    for dim in dimensions[i+1:]:  # Multiply by all subsequent dimension sizes
                        stride *= dim
                    offset += idx * stride  # Add this dimension's contribution to the offset

                # Seek to the exact byte position of the target element
                self.file.file.seek(data_start_pos + offset)
                # Read exactly one element's worth of binary data
                buffer = self.file.file.read(element_size)

                # Convert binary data to a NumPy array containing a single element
                single_element = np.frombuffer(buffer, dtype=dtype, count=1)

                # Handle endianness differences if needed
                if need_byteswap:
                    single_element = single_element.byteswap()

                # Extract the scalar value from the NumPy array
                result = single_element[0].item()

            # Case 2: Partial indexing - accessing a sub-array (slice of the array)
            # When fewer indices than dimensions are provided, we're extracting a sub-array
            else:
                # Calculate the shape of the resulting sub-array (remaining dimensions)
                result_shape = dimensions[len(indices):]  # Take dimensions that weren't indexed

                # Calculate the total number of elements in the sub-array
                total_elements = 1
                for dim in result_shape:
                    total_elements *= dim

                # Calculate the base offset to the start of the sub-array
                # This is similar to the scalar case, but only for the provided indices
                offset = 0
                for i, idx in enumerate(indices):
                    # Calculate stride for this dimension
                    stride = element_size
                    for dim in dimensions[i+1:]:
                        stride *= dim
                    offset += idx * stride

                # Read the entire sub-array as a contiguous block of memory
                # This is more efficient than reading elements individually
                self.file.file.seek(data_start_pos + offset)
                buffer = self.file.file.read(total_elements * element_size)

                # Convert the binary data to a NumPy array with the appropriate type
                result = np.frombuffer(buffer, dtype=dtype)

                # Reshape to match the sub-array dimensions
                if result_shape:
                    result = result.reshape(result_shape)

                # Handle endianness differences if needed
                if need_byteswap:
                    result = result.byteswap()

            return result
        else:
            # Object is a singular type (int, float, str, etc.) which doesn't support indexing
            # This includes primitive types like integers, floats, strings, etc.
            raise TypeError(f"Object of type '{symbol}' does not support indexing")

    def keys(self):
        """
        Return a list of keys from a dictionary object.

        Returns:
            list: List of keys from the dictionary

        Raises:
            TypeError: If the object is not a dictionary
        """
        # Store current position
        current_pos = self.reader._getPos()

        # Move to the position of this object
        self.reader._setPos(self.position)

        # Re-read the object type and handle footnotes
        is_footnote = True
        while is_footnote:
            symbol, size, dimensions = self.reader._read_header()
            if symbol == '*':
                # Skip over footnote without reading its content
                self.reader._read_object()
            else:
                is_footnote = False

        # Check if object is a dictionary
        if symbol != '{':
            self.reader._setPos(current_pos)
            raise TypeError(f"Object of type '{symbol}' is not a dictionary")

        # Read keys while skipping values
        keys = []
        while True:
            key_symbol, key_size, key_dimensions = self.reader._read_header()

            # Check if we've reached the end of the dictionary
            if key_symbol == '}':
                break

            # Read the key
            key = self.reader._read_element(key_symbol, key_size, key_dimensions)
            if isinstance(key, list):
                key = self.reader._convert_to_deep_tuple(key)

            keys.append(key)
            # Skip the value
            self._skip_object()

        # Restore original position
        self.reader._setPos(current_pos)
        return keys

    def __call__(self):
        """
        Convert the entire object to a Python object.

        Returns:
            Any: The Python object read from the file
        """
        # Save current position
        current_pos = self.reader._getPos()

        # Move to the position of this object
        self.reader._setPos(self.position)

        # Read the object
        result = self.reader._read_object()

        # Restore original position
        self.reader._setPos(current_pos)

        return result
