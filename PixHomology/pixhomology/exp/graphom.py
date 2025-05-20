import numpy as np
import ctypes
from pathlib import Path
import sys

# Define the Result structure in Python
class Graph(ctypes.Structure):
    _fields_ = [("edges", ctypes.POINTER(ctypes.c_int)),
                ("weights", ctypes.POINTER(ctypes.c_double))]

# Define the C PixHomology function
#print(Path(__file__).parent.parent)

#pixhom = np.ctypeslib.load_library('pixhomology', Path(__file__).parent.parent)
#pixhom = np.ctypeslib.load_library('libpixhom', Path(__file__).parent) 


# Load shared library
path = Path(__file__).parent.parent.parent
name = "graphom"
if sys.platform.startswith("win"):
    name = f"{name}.dll"
elif sys.platform.startswith("linux"):
    name = f"lib{name}.so"
elif sys.platform.startswith("darwin"):
    name = f"lib{name}.dylib"
else:
    raise ImportError(f"Your OS is not supported: {sys.platform}")

graphom = ctypes.CDLL(str(path / name))



graphom.computeGraph.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                                 ctypes.c_int,
                                 ctypes.c_int]
graphom.computeGraph.restype = Graph


# Define Python Wrapper
def image_to_graph(arr):
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
    result_struct = graphom.computeGraph(arr.astype(np.float64), num_rows, num_cols)
    edges = np.ctypeslib.as_array(result_struct.edges, shape=(num_rows, num_cols))
    weights = np.ctypeslib.as_array(result_struct.weights, shape=(num_rows, num_cols))

    return edges, weights


def graph_to_image(edges, weights):
    A = np.zeros((edges.size,edges.size))
    A[np.arange(edges.size), edges.flatten()] = 1
    A = np.eye(edges.size) - A
    A[np.eye(edges.size) == 1] = 1
    
    image = np.matmul(np.linalg.inv(A), -weights.flatten())
    image -= image.min()
    image = image.reshape(edges.shape)
    return image
