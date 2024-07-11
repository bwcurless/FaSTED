import numpy as np

# Calculates the expected values that should be in the D registers 
# for my mmaPTXTest in cuda

# The size of a single fragment
basic_shape = (8, 8)
a0 = np.arange(0, 64).reshape(basic_shape)
a1 = np.arange(64, 128).reshape(basic_shape)
a2 = np.arange(128, 192).reshape(basic_shape)
a3 = np.arange(192, 256).reshape(basic_shape)

a_full = np.hstack((np.vstack((a0, a1)), np.vstack((a2, a3))))

print(f"a0: {a0}")
print(f"a1: {a1}")
print(f"a2: {a2}")
print(f"a3: {a3}")

print(f"a_full:\n{a_full}")

b0 = np.arange(0, 64).reshape(basic_shape).T
b1 = np.arange(64, 128).reshape(basic_shape).T

b_full = np.vstack((b0, b1))

print(f"b0: {b0}")
print(f"b1: {b1}")
print(f"b_full:\n{b_full}")

d = a_full @ b_full

print(f"d:\n{d}")


