import numpy as np

def fit_plane_IRLS(points, iterations=10, epsilon=1e-6):
    # Prepare the design matrix X
    X = points[:, :3]
    ones = np.ones((X.shape[0], 1))
    X = np.hstack([X, ones])

    # Initialize coefficients and weights
    plane_coefficients = np.zeros(4)
    weights = np.ones(X.shape[0])

    for _ in range(iterations):
        W = np.diag(weights)
        # Weighted least squares solution
        plane_coefficients = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ points[:, 2])
        
        # Compute residuals and update weights
        residuals = points[:, 2] - X @ plane_coefficients
        weights = 1.0 / (np.abs(residuals) + epsilon)
        
    normal_vector = plane_coefficients[:3]
    d = plane_coefficients[3]
    
    return normal_vector, d