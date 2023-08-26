import numpy as np

def random_three_points(points):
    ids = np.random.choice(points.shape[0], 3, replace=False)
    return points[ids]

def fit_plane(points):
    p1, p2, p3 = points
    normal_vector = np.cross(p2 - p1, p3 - p1)
    d = np.dot(normal_vector, p1)
    return normal_vector, d

def compute_error(plane, points):
    normal_vector, d = plane
    errors = np.abs(np.dot(points, normal_vector) - d) / np.linalg.norm(normal_vector)
    return errors

def least_median_squares(points, iterations):
    best_plane = None
    best_median = float('inf')

    for _ in range(iterations):
        sample_points = random_three_points(points)
        candidate_plane = fit_plane(sample_points)
        errors = compute_error(candidate_plane, points)
        median_error = np.median(errors ** 2)

        if median_error < best_median:
            best_median = median_error
            best_plane = candidate_plane

    # Once the best model is found, we can find the inliers using a threshold (optional)
    errors = compute_error(best_plane, points)
    inlier_mask = errors ** 2 < best_median
    best_inliers = points[inlier_mask]

    return best_plane, best_inliers
