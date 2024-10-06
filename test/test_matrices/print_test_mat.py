import argparse
import numpy as np

def read_mm_file(filename):
    """
    Reads a matrix from a Matrix Market file (.mtx format)
    
    :param filename: Path to the Matrix Market file
    :return: Numpy array representing the matrix
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip comments and MatrixMarket header
    header_skipped = False
    matrix_data = []
    for line in lines:
        # Ignore comment lines and the header
        if line.startswith('%'):
            continue
        if not header_skipped:
            header_skipped = True
        # Add the matrix data (coordinates and value for sparse)
        matrix_data.append(line.strip().split())


    # Read matrix size from the first line of matrix data (after the header)
    rows, cols, _ = map(int, matrix_data[0])
    matrix_data = matrix_data[1:]  # Remove size from the data

    # Initialize a zero matrix
    matrix = np.zeros((rows, cols))

    # Fill the matrix with the non-zero elements
    for data in matrix_data:
        row, col, value = int(data[0]) - 1, int(data[1]) - 1, float(data[2])  # Convert 1-based to 0-based index
        matrix[row, col] = value

    return matrix

def print_matrix(matrix):
    """
    Prints the matrix in a nice, easy-to-read format
    
    :param matrix: Numpy array representing the matrix
    """
    if matrix is not None:
        print("Matrix content:")
        for row in matrix:
            print(" ".join(f"{value:8.2f}" for value in row))
    else:
        print("Matrix is None, cannot print.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and print a matrix from a Matrix Market (.mtx) file")
    parser.add_argument('--file', type=str, required=True, help="Path to the Matrix Market file (.mtx)")
    
    args = parser.parse_args()
    filename = args.file
    
    matrix = read_mm_file(filename)
    
    if matrix is not None:
        print(f"Matrix dimensions: {matrix.shape[0]} x {matrix.shape[1]}")
    print_matrix(matrix)

