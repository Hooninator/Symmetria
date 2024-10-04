import argparse
import numpy as np

def generate_matrix_market(size, non_zeros):
    """Generate a sparse matrix in Matrix Market format."""
    # Generate a random sparse matrix with specified size and non-zero elements
    matrix = np.zeros((size, size), dtype=int)
    non_zero_positions = np.random.choice(size * size, non_zeros, replace=False)
    for pos in non_zero_positions:
        row = pos // size
        col = pos % size
        matrix[row, col] = np.random.randint(1, 10)  # Random non-zero values between 1 and 9

    # Convert matrix to Matrix Market format
    rows, cols = np.nonzero(matrix)
    data = matrix[rows, cols]

    mm_format = "%%MatrixMarket matrix coordinate integer general\n%d %d %d\n" % (size, size, non_zeros)
    mm_format += "\n".join(f"{row + 1} {col + 1} {value}" for row, col, value in zip(rows, cols, data))
    mm_format += "\n"

    tuples = [(col, row, value) for col, row, value in zip(cols, rows, data)]
    tuples.sort(key=lambda a: a[0])
    mm_format_transpose = "%%MatrixMarket matrix coordinate integer general\n%d %d %d\n" % (size, size, non_zeros)
    mm_format_transpose += "\n".join(f"{col + 1} {row + 1} {value}" for col, row, value in tuples)
    mm_format_transpose += "\n"

    matrix_transpose = np.transpose(matrix)
    matrix_product = np.matmul(matrix, matrix_transpose)

    rows, cols = np.nonzero(matrix_product)
    data = matrix_product[rows, cols]
    mm_product = "%%MatrixMarket matrix coordinate integer general\n%d %d %d\n" % (size, size, non_zeros)
    mm_product += "\n".join(f"{row + 1} {col + 1} {value}" for row, col, value in zip(rows, cols, data))
    mm_product += "\n"

    return mm_format, mm_format_transpose, mm_product

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate a sparse matrix in Matrix Market format.")
    parser.add_argument("--size", type=int, default=16, help="Matrix dimension (NxN)")
    parser.add_argument("--non_zeros", type=int, default=16, help="Number of non-zero elements")
    parser.add_argument("--name", type=str, help="Name of file to store matrix in")
    args = parser.parse_args()

    # Generate and output the matrix in Matrix Market format
    mm_format, mm_format_transpose, mm_product = generate_matrix_market(args.size, args.non_zeros)
    with open(args.name+".mtx", 'w') as file:
        file.write(mm_format)
    with open(args.name+"_transpose.mtx", 'w') as file:
        file.write(mm_format_transpose)
    with open(args.name+"_product.mtx", 'w') as file:
        file.write(mm_product)

# The script will execute the main function if run as a standalone script
if __name__ == "__main__":
    main()

