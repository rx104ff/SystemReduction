import numpy as np

# Example matrices
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

t = 0

# Define a custom function that uses corresponding elements from both matrices
def custom_function(x, y, t):
    return x + y + x * y * t  # Example: Combine elements with an arbitrary operation

# Vectorized application using NumPy
result = custom_function(matrix1, matrix2, t)
matrix1[0,0] = 32

print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Resulting Matrix:\n", result)
