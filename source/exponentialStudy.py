#!/usr/bin/python3

import ctypes

# SimSearch::Results runFromExponentialDataset(int size, int dimensionality, double lambda,
#                                             double mean, double epsilon);
#struct Results {
#    double totalExecutionTime;   // How long it took to execute the routine
#    Mma::mmaShape problemShape;  // The actual shape of the problem, m x n x k
#};

#struct mmaShape {
#    int m{};
#    int n{};
#    int k{};
#};

class mmaShape(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int),
    ]

# The result types
class Results(ctypes.Structure):
    _fields_ = [
        ("totalExecutionTime", ctypes.c_double),
        ("problemShape", mmaShape),
    ]


# Load the shared library
findPairs = ctypes.CDLL("./main.so")  

# Define the function prototype
findPairs.runFromExponentialDataset.restype = Results
findPairs.runFromExponentialDataset.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]

# Call the C function
result = findPairs.runFromExponentialDataset(100, 17, 40, 0, 0.03)

# Access the struct members
print("totalExecutionTime:", result.totalExecutionTime)

