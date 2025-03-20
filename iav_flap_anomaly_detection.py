import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

def make_data(n_train=10000, n_test=10000):
    """
    Create the dataset.
    
    Parameters
    ----------
    
    n_train: int
        The number of training data points. Default: 10000
    n_test: int
        The number of test data points. Default: 10000
    
    Returns
    -------
    X_train: array_like
        Training data.
    X_test: array_like
        Test data.
    test_ground_truth_labels: array_like
        Ground truth labels for the test data.
    """
    n_normal = n_train + n_test//2
    n_anomaly = n_train + n_test - n_normal
    
    rng = np.random.default_rng(42)
    
    X, y = make_moons(n_samples=(n_normal, n_anomaly), noise=0.05, random_state=23, shuffle=False)
    y_normalized = np.where(y==0, 1, -1) # Adapt labels to OneClassSVM labels

    X_normal = rng.permutation(X[:n_normal, :], axis=0), y_normalized[:n_normal]
    X_anomaly, y_anomaly = X[n_normal:, :], y_normalized[n_normal:]

    X_normal, y_normal = rng.permutation(X[:n_normal, :], axis=0), y_normalized[:n_normal]
    X_anomaly, y_anomaly = X[n_normal:, :], y_normalized[n_normal:]
    
    X_train = X_normal[:n_train, :]

    X_test = np.vstack([X_normal[n_train:, :], X_anomaly])
    y_test = np.hstack([y_normalized[n_train:], y_anomaly])

    shuffled_indices = np.arange(X_test.shape[0], dtype=np.uint64)
    rng.shuffle(shuffled_indices)
    X_test = X_test[shuffled_indices, :]
    test_ground_truth_labels = y_test[shuffled_indices]
    
    return X_train, X_test, test_ground_truth_labels

def plot_data(X_train, X_test, test_ground_truth):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(12.8,4.8))
    ax1.scatter(X_train[:, 0], X_train[:, 1], label="Training data")
    ax1.set_xlim(-1.2, 2.2)
    ax1.set_ylim(-0.7, 1.2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Training data")
    ax1.legend()
    ax2.scatter(X_train[:, 0], X_train[:, 1], label="Training data (ok)", alpha=0.7)
    ax2.scatter(X_test[:, 0], X_test[:, 1], label="Test data (ok + not ok)", alpha=0.7)
    ax2.set_xlim(-1.2, 2.2)
    ax2.set_ylim(-0.7, 1.2)
    ax2.set_title("Training vs. test data")
    ax2.legend()
    
    inlier_mask = test_ground_truth == 1
    outlier_mask = np.logical_not(inlier_mask)
    
    ax3.scatter(X_test[inlier_mask, 0], X_test[inlier_mask, 1], label="Ok data", alpha=0.7)
    ax3.scatter(X_test[outlier_mask, 0], X_test[outlier_mask, 1], label="Not ok data", alpha=0.7)
    ax3.set_xlim(-1.2, 2.2)
    ax3.set_ylim(-0.7, 1.2)
    ax3.set_title("What detection should ideally look like")
    ax3.legend()
    plt.show()


def plot_results(model, X_test, y_pred, 
                 x_min=-1.2, x_max=2.2, 
                 y_min=-0.7, y_max=1.2):
    """
    Plot the final detection results of a trained One-Class SVM.

    Parameters
    ----------
    model : object
        A trained model that implements the .decision_function() method, 
        e.g. One-Class SVM or other anomaly detection estimators.
    X_test : ndarray of shape (n_samples, 2)
        The test dataset.
    y_pred : ndarray of shape (n_samples,)
        Predicted labels for the test set (1 for inliers, -1 for outliers).
    x_min, x_max : float
        The minimum and maximum x-values for the plot's axis and contour.
    y_min, y_max : float
        The minimum and maximum y-values for the plot's axis and contour.
    """
    
    # Create a meshgrid for contour plotting
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Compute the decision function for each point in the grid
    Z = model.decision_function(grid)
    Z = Z.reshape(xx.shape)

    # Plot the decision function using contour
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50),
                 cmap=plt.cm.RdBu_r, alpha=0.6)

    # Plot the test data points
    plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1],
                c='blue', s=5, label='Okay Data')
    plt.scatter(X_test[y_pred != 1, 0], X_test[y_pred != 1, 1],
                c='red', s=5, label='Not Okay Data')

    # Set the axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Set titles and labels
    plt.title("One-Class SVM Anomaly Detection")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
