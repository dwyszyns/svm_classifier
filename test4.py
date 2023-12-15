import csv
import numpy as np
import matplotlib.pyplot as plt

# kernel functions
def kernel_polynomial(a, b, degree=3):
    return pow((np.dot(a, b) + 1), degree)

def kernel_linear(a, b):
    return np.dot(a, b)

def kernel_radial(a, b, gamma=1):
    return np.exp(-pow(np.linalg.norm(a - b), 2) / (2 * pow(gamma, 2)))

class SupportVectorMachine:
    def __init__(self, kernel=kernel_linear, C=None):
        self.kernel = kernel
        self.C = C
        self.support_vectors = None
        self.support_vectors_y = None
        self.alphas = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Calculate kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # Quadratic programming to find alphas
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        if self.C is not None:
            G = np.vstack((np.eye(n_samples), -np.eye(n_samples)))
            h = np.hstack((self.C * np.ones(n_samples), np.zeros(n_samples)))
        else:
            G = -np.eye(n_samples)
            h = np.zeros(n_samples)

        A = y.reshape(1, -1)
        b = np.array([0.0])

        solution = self.solve_qp(P, q, G, h, A, b)
        self.alphas = solution['x']

        # Find support vectors
        sv_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[sv_indices]
        self.support_vectors_y = y[sv_indices]

        # Calculate bias
        self.bias = np.mean(self.support_vectors_y - self.decision_function(self.support_vectors))

    def decision_function(self, X):
        n_samples = X.shape[0]
        result = np.zeros(n_samples)
        for i in range(n_samples):
            result[i] = np.sum(self.alphas * self.support_vectors_y * np.array([self.kernel(X[i], x) for x in self.support_vectors])) + self.bias
        return result

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def solve_qp(self, P, q, G, h, A, b):
        n = P.shape[0]
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        solution = qp(P, q, G, h, A, b)
        return solution

def read_csv():
    xs = []
    ys = []
    with open('red.csv', 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=';')
        next(reader)  # Skip header
        for row in reader:
            xs.append(np.array(row[:11], dtype=float))
            ys.append(1.0 if int(row[-1]) >= 6 else -1.0)
    return np.array(xs), np.array(ys)

def calculate(svm, x_training, y_training, x, y):
    svm.fit(x_training, y_training)
    predictions = svm.predict(x)
    correct = np.sum(predictions == y)
    return correct, predictions

def output_result(correct, predictions):
    print(f"Correctly guessed {correct} in {len(predictions)} tries")
    ratio = correct / len(predictions)
    print(ratio)
    return ratio

def get_data(last=1000):
    x1, y1 = read_csv()
    x_training = x1[:last]
    y_training = y1[:last]
    x = x1[last:]
    y = y1[last:]
    return x_training, y_training, x, y

def hard_margin(kernel=kernel_linear, number=1000):
    svm_algorithm = SupportVectorMachine(kernel=kernel)
    return output_result(*calculate(svm_algorithm, *get_data(number)))

def soft_margin(kernel=kernel_linear, C=None, number=1000):
    svm_algorithm = SupportVectorMachine(kernel=kernel, C=C)
    return output_result(*calculate(svm_algorithm, *get_data(number)))

def main():
    x = []
    y = []
    plt.xlabel('C')
    plt.ylabel('effectiveness')
    plt.title('C in RBF (gamma=100)')
    for i in range(-4, 6):
        y.append(soft_margin(kernel=kernel_radial, C=pow(10, i)))
        x.append("10^"+str(i))
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()
