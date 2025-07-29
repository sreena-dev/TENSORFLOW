import numpy as np

def dense(W, b, x):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, x) + b[j]
        a_out[j] = g(z)
    return a_out

def g(z):
    sigma = 1 / (1 + (np.exp(-z)))
    return sigma

def sequential(x, W1, b1, W2, b2, W3, b3):
    a_1 = dense(W1, b1, x)
    a_2 = dense(W2, b2, a_1)
    a_3 = dense(W3, b3, a_2)
    f_x = a_3
    return f_x

# Adjusted shapes for single input x = np.array([1, 1])
W1 = np.array([[-3, 4],
               [ 1, 2]])  # 2 input features, 2 units
b1 = np.array([1, 2])      # 2 biases

W2 = np.array([[5, -6],
               [1, 2]])  # 2 input features (from previous layer), 2 units
b2 = np.array([1, 2])      # 2 biases

W3 = np.array([[1, 2],
               [1, 2]])  # 2 input features, 2 units
b3 = np.array([1, 2])      # 2 biases

x = np.array([0, 0])

def main():
    print(sequential(x, W1, b1, W2, b2, W3, b3))

if __name__ == "__main__":
    main()