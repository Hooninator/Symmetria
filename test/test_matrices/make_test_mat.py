import argparse
import numpy as np

def generate_matrix_market(nrows, ncols, non_zeros):
    """Generate a sparse matrix in Matrix Market format."""
    # Generate a random sparse matrix with specified size and non-zero elements
    matrix = np.zeros((nrows, ncols), dtype=int)
    non_zero_positions = np.random.choice(nrows*ncols, non_zeros, replace=False)
    for pos in non_zero_positions:
        row = pos // nrows 
        col = pos % ncols 
        matrix[row, col] = np.random.randint(1, 10)  # Random non-zero values between 1 and 9

    # Convert matrix to Matrix Market format
    rows, cols = np.nonzero(matrix)
    data = matrix[rows, cols]

    mm_format = "%%MatrixMarket matrix coordinate integer general\n%d %d %d\n" % (nrows, ncols, non_zeros)
    mm_format += "\n".join(f"{row + 1} {col + 1} {value}" for row, col, value in zip(rows, cols, data))
    mm_format += "\n"

    tuples = [(col, row, value) for col, row, value in zip(cols, rows, data)]
    tuples.sort(key=lambda a: a[0])
    mm_format_transpose = "%%MatrixMarket matrix coordinate integer general\n%d %d %d\n" % (nrows, ncols, non_zeros)
    mm_format_transpose += "\n".join(f"{col + 1} {row + 1} {value}" for col, row, value in tuples)
    mm_format_transpose += "\n"

    matrix_transpose = np.transpose(matrix)
    matrix_product = np.matmul(matrix, matrix_transpose)

    rows, cols = (np.nonzero((matrix_product)))

    data = matrix_product[rows, cols]
    mm_product = "%%MatrixMarket matrix coordinate integer general\n%d %d %d\n" % (nrows, nrows, non_zeros)
    mm_product += "\n".join(f"{row + 1} {col + 1} {value}" for row, col, value in zip(rows, cols, data))
    mm_product += "\n"

    return mm_format, mm_format_transpose, mm_product

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate a sparse matrix in Matrix Market format.")
    parser.add_argument("--rows", type=int, default=16, help="Matrix dimension (NxN)")
    parser.add_argument("--cols", type=int, default=16, help="Matrix dimension (NxN)")
    parser.add_argument("--non_zeros", type=int, default=16, help="Number of non-zero elements")
    parser.add_argument("--name", type=str, help="Name of file to store matrix in")
    args = parser.parse_args()

    # Generate and output the matrix in Matrix Market format
    mm_format, mm_format_transpose, mm_product = generate_matrix_market(args.rows, args.cols, args.non_zeros)
    with open(args.name+".mtx", 'w') as file:
        file.write(mm_format)

    #with open(args.name+"_transpose.mtx", 'w') as file:
    #    file.write(mm_format_transpose)

    with open(args.name+"_product.mtx", 'w') as file:
        file.write(mm_product)

if __name__ == "__main__":
    main()

