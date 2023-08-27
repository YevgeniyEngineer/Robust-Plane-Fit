import numpy as np

def fit_plane_m_estimator(points, iterations=100, threshold=1.0, alpha=0.01):
    num_points = points.shape[0]
    
    # Initialize the plane parameters (a, b, c, d) for the equation ax + by + cz = d
    params = np.array([0.0, 0.0, 1.0, 0.0])
    
    for _ in range(iterations):
        a, b, c, d = params
        
        # Calculate the residuals
        residuals = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] - d
        
        # Calculate Huber weights for each point
        huber_weights = np.ones_like(residuals)
        mask = np.abs(residuals) > threshold
        huber_weights[mask] = threshold / np.abs(residuals[mask])
        
        # Construct the weighted design matrix
        A = np.column_stack([points, np.ones(num_points)])
        W = np.diag(huber_weights)
        
        # Update the parameters
        params = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ points[:, 2], rcond=None)[0]
        
    # Extract the plane parameters
    normal_vector = params[:3]
    d = params[3]
    
    return normal_vector, d
