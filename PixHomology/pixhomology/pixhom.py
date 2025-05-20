import numpy as np
import ctypes
from pathlib import Path
import sys

# Define the Result structure in Python
class Result(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)),
                ("length", ctypes.c_int)]

# Define the C PixHomology function
#print(Path(__file__).parent.parent)

#pixhom = np.ctypeslib.load_library('pixhomology', Path(__file__).parent.parent)
#pixhom = np.ctypeslib.load_library('libpixhom', Path(__file__).parent) 


# Load shared library
path = Path(__file__).parent.parent
name = "pixhom"
if sys.platform.startswith("win"):
    name = f"{name}.dll"
elif sys.platform.startswith("linux"):
    name = f"lib{name}.so"
elif sys.platform.startswith("darwin"):
    name = f"lib{name}.dylib"
else:
    raise ImportError(f"Your OS is not supported: {sys.platform}")

pixhom = ctypes.CDLL(str(path / name))



pixhom.computePH.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                               ctypes.c_int,
                               ctypes.c_int]
pixhom.computePH.restype = Result


# Define Python Wrapper
def computePH(arr):
    # Check if the input is a NumPy array
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    # Check if the array is 2-dimensional
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    # Check if the array is float32
    #if arr.dtype != np.float32:
    #    raise ValueError("Input array must have dtype 'float32'")

    # Get the size of the array
    num_rows, num_cols = arr.shape

    # Call the C function
    result_struct = pixhom.computePH(arr.astype(np.float64), num_rows, num_cols)
    result = np.ctypeslib.as_array(result_struct.data, shape=(int(result_struct.length/2), 2))

    return result
