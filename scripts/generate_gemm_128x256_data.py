import numpy as np

def main():
  # Generate two 128x256 matrices of random numbers
  matrix_a = np.random.rand(128, 128)
  matrix_b = np.random.rand(128, 128)

  # Perform matrix multiplication
  matrix_c = np.dot(matrix_a, matrix_b)

  # Save matrices to files
  np.savetxt('/tmp/matrix_a_128x256.csv', matrix_a, delimiter=',')
  np.savetxt('/tmp/matrix_b_128x256.csv', matrix_b, delimiter=',')
  np.savetxt('/tmp/matrix_c_128x256.csv', matrix_c, delimiter=',')


if __name__ == '__main__':
  main()