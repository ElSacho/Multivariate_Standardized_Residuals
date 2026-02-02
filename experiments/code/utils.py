import pandas as pd
import random
import numpy as np
import torch

def calculate_accuracy(model, dataloader):
    """
    Computes the model's accuracy on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The DataLoader for the dataset.

    Returns
    -------
    accuracy : float
        The model's accuracy on the dataset.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            _, true = torch.max(y, 1)
            total += y.size(0)
            correct += (predicted == true).sum().item()

    accuracy = 100 * correct / total
    return accuracy



class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        Initialize the KMeans class.

        Parameters:
        - n_clusters (int): Number of clusters to form.
        - max_iter (int): Maximum number of iterations of the algorithm.
        - tol (float): Tolerance for convergence. The algorithm stops if the centroids change less than this.
        - random_state (int): Seed for random number generator (for reproducibility).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """
        Fit the KMeans model to the data.

        Parameters:
        - X (numpy.ndarray): Data to cluster, of shape (n_samples, n_features).
        """
        np.random.seed(self.random_state)
        # Randomly initialize the centroids
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # Assign each point to the nearest centroid
            labels = self._assign_clusters(X)

            # Compute new centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Parameters:
        - X (numpy.ndarray): Data to predict, of shape (n_samples, n_features).

        Returns:
        - labels (numpy.ndarray): Index of the cluster each sample belongs to.
        """
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        """
        Assign each point in X to the nearest centroid.

        Parameters:
        - X (numpy.ndarray): Data to cluster, of shape (n_samples, n_features).

        Returns:
        - labels (numpy.ndarray): Index of the cluster each sample belongs to.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


def print_results(results):
    df_results = pd.DataFrame(results)

    # Round 'Logdet' and 'Non relaxed problem cost' values to two significant digits
    df_results["Logdet"] = df_results["Logdet"].apply(lambda x: f"{x:.3g}")
    df_results["Non relaxed problem cost"] = df_results["Non relaxed problem cost"].apply(lambda x: f"{x:.3g}")

    # Mask duplicates in the "Perturbation" column
    df_results.loc[df_results.duplicated(subset=["Perturbation"]), "Perturbation"] = ""

    # Function to add double borders between perturbations
    def add_double_border(s):
        border_style = "border-top: 3px double black;"
        return [border_style if isinstance(s.index[i], int) and i > 0 and df_results["Perturbation"].iloc[i] == "" else "" for i in range(len(s))]

    # Apply styles to the DataFrame
    styled_df = (
        df_results.style
        .apply(add_double_border, axis=1)
    )
    # Display the styled DataFrame
    return styled_df

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  


def add_nan(y, min_nan=1, max_nan=None):
    """
    Randomly sets several elements per row in y to NaN.
    
    Parameters:
        y (np.ndarray): A NumPy array of shape (n, k).
        min_nan (int): Minimum number of NaNs per row.
        max_nan (int): Maximum number of NaNs per row. If None, defaults to k.
        
    Returns:
        np.ndarray: A copy of y with NaNs inserted.
    """
    y_nan = y.copy()
    n, k = y_nan.shape
    if max_nan is None:
        max_nan = k-1

    for i in range(n):
        num_nan = np.random.randint(min_nan, max_nan + 1)  # number of NaNs for this row
        nan_indices = np.random.choice(k, size=num_nan, replace=False)
        y_nan[i, nan_indices] = np.nan
        
    return y_nan


def add_nan_extreme_values(y, percentage=0.1):
    """
    Set NaN on the most extreme values of y (by absolute value).

    Parameters:
        y (np.ndarray): Array of shape (n, k).
        percentage (float): Fraction of extreme values to replace with NaN (e.g. 0.1 = 10%).

    Returns:
        np.ndarray: Copy of y with NaNs inserted.
    """
    y_nan = y.copy()

    # Flatten to compute global threshold
    flat = np.abs(y_nan).ravel()
    
    # Number of values to replace
    n_extreme = int(np.ceil(percentage * flat.size))
    if n_extreme == 0:
        return y_nan

    # Threshold for extreme values
    threshold = np.partition(flat, -n_extreme)[-n_extreme]

    # Mask extreme values
    mask = np.abs(y_nan) >= threshold
    y_nan[mask] = np.nan

    return y_nan
