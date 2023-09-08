import numpy as np

from utils import load_dataset, problem


class RidgeRegression:


    def __init__(self, degree=2, alpha=1e-4):
        super(RidgeRegression, self).__init__(degree)
        self.alpha = alpha
        self.degree: int = degree

    
    def train(self, x: np.ndarray, y: np.ndarray, _lambda: float) -> np.ndarray:
        x_polly = self.polyfeatures(x,self.degree)
        x_standarized = (x_polly - x_polly.mean(axis=0)) / x_polly.std(axis=0)
        x_encoded = np.column_stack([np.ones(X.shape[0]), x_standarized])
        regularization_matrix = self.alpha * np.eye(x_encoded.shape[1])
        regularization_matrix[0, 0] = 0
        #print(regularization_matrix)
        
        self.theta = np.linalg.pinv(x_encoded.T @ x_encoded + regularization_matrix) @ x_encoded.T @ y



def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x_polly = self.make_polynomial(X)
        x_standarized = (x_polly - x_polly.mean(axis=0)) / x_polly.std(axis=0)
        x_encoded = np.column_stack([np.ones(X.shape[0]), x_standarized])
        return x_encoded @ self.theta


def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        n = len(X)
        result = np.empty((n, degree))

        for i in range(0, n):
            x = X[i]
            for j in range(0, degree):
                result[i, j] = x ** (j + 1)

        return result



def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encode a vector `y`.
    One hot encoding takes an array of integers and coverts them into binary format.
    Each number i is converted into a vector of zeros (of size num_classes), with exception of i^th element which is 1.

    Args:
        y (np.ndarray): An array of integers [0, num_classes), of shape (n,)
        num_classes (int): Number of classes in y.

    Returns:
        np.ndarray: Array of shape (n, num_classes).
        One-hot representation of y (see below for example).

    Example:
        ```python
        > one_hot([2, 3, 1, 0], 4)
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        ```
    """
    if isinstance(y, list):
        yt = np.asarray(y)
    else:
        yt = y
    if not len(yt.shape) == 1:
        raise AttributeError('y array must be 1-dimensional')
    if num_classes == 'auto':
        # uniq = np.unique(yt).shape[0]
        uniq = np.max(yt + 1)
    else:
        uniq = num_classes
    if uniq == 1:
        ary = np.array([[0.]])

    else:
        ary = np.zeros((len(y), uniq))
        for i, val in enumerate(y):
            ary[i, val] = 1

    return ary


def main():

    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    # Convert to one-hot
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)

    _lambda = 1e-4

    w_hat = train(x_train, y_train_one_hot, _lambda)

    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)

    print("Ridge Regression Problem")
    print(
        f"\tTrain Error: {np.average(1 - np.equal(y_train_pred, y_train)) * 100:.2g}%"
    )
    print(f"\tTest Error:  {np.average(1 - np.equal(y_test_pred, y_test)) * 100:.2g}%")


if __name__ == "__main__":
    main()
