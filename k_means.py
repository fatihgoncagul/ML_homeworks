import numpy as np

from utils import problem



@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    m, n = data.shape
    centroids = np.zeros((num_centers, n))
    
        
    for k in range(num_centers):   
            points = data[classifications == k]  # to get a list of all data points in data assigned to the centroid  
            centroids[k] = np.mean(points, axis = 0) # to compute the mean of the points assigned
            
    return centroids 


@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to cegnters given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
            # Set K
    K = centers.shape[0]
    
       
    idx = np.zeros(data.shape[0], dtype=int)
    
        
    for i in range(data.shape[0]):
            # Array to hold distance between X[i] and each centroids[j]
        distance = [] 
        for j in range(centers.shape[0]):
            norm_ij = np.linalg.norm(data[i] - centers[j]) # the norm between (X[i] - centroids[j])
            distance.append(norm_ij)
            
        idx[i] = np.argmin(distance) # index of minimum value in distance
    return idx


@problem.tag("hw4-A")
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    n = data.shape[0]
    k = centers.shape[0]

    nearest_center = cluster_data(data, centers)

    se = 0
    for data_v, center_idx in zip(data,nearest_center):
        center = centers[int(center_idx)]
        distance = np.linalg.norm(data_v - center)
        se += distance

    mse = se/n

    return mse


@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> np.ndarray:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing trained centers.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(data.shape[0])
    # Take the first K examples as centroids
    centroids = data[randidx[:num_centers]]
    classifications = np.zeros(data.shape[0], dtype=np.int64)

    loss = 0
    max_iter = 50

    for m in range(0, max_iter):
        

        classifications = cluster_data(data, centroids)
        new_centroids = calculate_centers(data, classifications, num_centers)
        new_loss = calculate_error(data, new_centroids)
        obj_func = np.abs(centroids - new_centroids)
        #Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon
        if np.max(obj_func) < epsilon:
            return new_centroids

        centroids = new_centroids
        loss = new_loss

        print(loss)


    print("Failed to converge!")

    return centroids
