import numpy as np

def random_three_points(points):
    ids = np.random.choice(points.shape[0], 3, replace=False)
    return points[ids]

def fit_plane(points):
    p1, p2, p3 = points
    normal_vector = np.cross(p2 - p1, p3 - p1)
    d = np.dot(normal_vector, p1)
    return normal_vector, d

def theil_sen_estimator(points, iterations):
    normals = []
    ds = []
    for _ in range(iterations):
        sample_points = random_three_points(points)
        normal, d = fit_plane(sample_points)
        normals.append(normal)
        ds.append(d)
    
    normals = np.array(normals)
    ds = np.array(ds)
    
    median_normal = np.median(normals, axis=0)
    median_d = np.median(ds)
    
    return median_normal, median_d