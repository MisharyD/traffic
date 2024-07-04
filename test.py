import numpy as np

# Create two 2D arrays
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])

# Horizontal stacking
result = np.vstack((array1, array2))

print("Array 1:\n", array1)
print("Array 2:\n", array2)
print("Horizontally Stacked Array:\n", result)
