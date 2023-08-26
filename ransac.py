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

def ransac(points, threshold, iterations):
    best_plane = None
    best_inliers = None
    max_inliers = 0

    for _ in range(iterations):
        sample_points = random_three_points(points)
        candidate_plane = fit_plane(sample_points)
        errors = compute_error(candidate_plane, points)
        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_plane = candidate_plane
            best_inliers = points[inlier_mask]

    return best_plane, best_inliers