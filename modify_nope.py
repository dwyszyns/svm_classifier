import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SVM:
    def __init__(self, learning_rate=0.1, lambda_param=0.01, max_iterations=100, kernel='linear', degree=2, gamma=0.1):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.theta = None

    def fit(self, X, y):
        # Add a column of ones for the bias term
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # Initialize weights
        self.theta = np.zeros(X_bias.shape[1])
        print(y.iloc[0])
        print(type(X))
        # Fit the model using stochastic gradient descent
        for iteration in range(self.max_iterations):
            for i in range(X_bias.shape[0]):
                # Use random choice instead of randint for proper sampling
                random_index = np.random.choice(X_bias.shape[0])

                xi = X_bias[random_index, :]
                yi = y.iloc[random_index]

                # Compute the gradient of the loss function
                gradient = self.gradient(xi, yi)

                # Update weights
                self.theta = self.theta - self.learning_rate * gradient

    def gradient(self, xi, yi):
        # Compute the gradient of the loss function
        if self.kernel == 'linear':
            hinge_loss = 1 - yi * np.dot(xi, self.theta)
            gradient = -yi * xi if hinge_loss > 0 else 2 * self.lambda_param * self.theta
        elif self.kernel == 'poly':
            hinge_loss = 1 - yi * self.decision_function(xi)
            gradient = -yi * xi * hinge_loss if hinge_loss > 0 else 2 * self.lambda_param * self.theta
        elif self.kernel == 'rbf':
            hinge_loss = 1 - yi * self.decision_function(xi)
            gradient = -yi * xi * hinge_loss if hinge_loss > 0 else 2 * self.lambda_param * self.theta
        else:
            raise ValueError("Unsupported kernel type")

        return gradient

    def decision_function(self, xi):
        # Decision function for different kernel types
        if self.kernel == 'linear':
            return np.dot(xi[1:], self.theta[1:]) + self.theta[0]
        elif self.kernel == 'poly':
            return (np.dot(xi[1:], self.theta[1:]) + self.theta[0]) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(xi[1:] - self.theta[1:]) ** 2)
        else:
            raise ValueError("Unsupported kernel type")

    def predict(self, X):
        # Predictions
        X_bias = np.c_[np.ones(X.shape[0]), X]
        predictions = np.array([1 if self.decision_function(xi) >= 0 else -1 for xi in X_bias])
        return predictions


# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Discretize the target variable
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else -1)

# Split data into training and testing sets
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implement SVM with RBF kernel
svm_model_rbf = SVM(kernel='poly', gamma=0.1)
svm_model_rbf.fit(X_train, y_train)

# Evaluate model accuracy
predictions_rbf = svm_model_rbf.predict(X_test)
accuracy_rbf = np.mean(predictions_rbf == y_test)
print(f'Accuracy for SVM with RBF kernel: {accuracy_rbf}')
