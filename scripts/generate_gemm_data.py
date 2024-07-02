import numpy as np

def main():
  # Generate two 256x512 matrices of random numbers
  matrix_a = np.random.rand(128, 256)
  matrix_b = np.random.rand(256, 128)

  # Perform matrix multiplication
  matrix_c = np.dot(matrix_a, matrix_b)

  # Save matrices to files
  np.savetxt('/tmp/matrix_a.csv', matrix_a, delimiter=',')
  np.savetxt('/tmp/matrix_b.csv', matrix_b, delimiter=',')
  np.savetxt('/tmp/matrix_c.csv', matrix_c, delimiter=',')


if __name__ == '__main__':
  main()