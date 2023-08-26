from ransac import ransac
from least_median_of_squares import least_median_squares
from iteratively_reweighted_least_squares import fit_plane_IRLS
from m_estimator import fit_plane_m_estimator
from theil_sen_estimator import theil_sen_estimator

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NUMBER_OF_POINTS = 200
NUMBER_OF_OUTLIERS = 60

THRESHOLD_RANSAC = 0.05
NUMBER_OF_ITERATIONS_RANSAC = 50

NUMBER_OF_ITERATIONS_LMEDS = 50

THRESHOLD_M_ESTIMATOR = 1
ALPHA_M_ESTIMATOR = 0.01
NUMBER_OF_ITERATIONS_M_ESTIMATOR = 10

EPSILON_IRLS = 0.000001
NUMBER_OF_ITERATIONS_IRLS = 5

NUMBER_OF_ITERATIONS_THEIL_SEN = 100

def plot_results(points_with_outliers, a, b, c, d, title):

    # Plot the points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_with_outliers[:, 0], points_with_outliers[:, 1], points_with_outliers[:, 2], c='r', marker='o')

    # Plot the fitted plane
    xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    zz = (d - a * xx - b * yy) / c

    ax.plot_surface(xx, yy, zz, color='g', alpha=0.5)

    plt.title(title)
    plt.show()

if __name__ == "__main__":
    """
        Time complexity:
        K - number of iterations
        N - number of data points
        M - number of parameters estimated

        0. Theil-Sen Estimator: O(K x M)
        1. Iteratively Re-weighted Least Squares: O(K x N x M x M)
        2. M-Estimator: O(K x N x M x M)
        3. Random Sample Consensus: O(K x N x (M + M x M))
        4. Least Median Of Squares: O(K x N x (M + log(N))

        Robustness to Outliers:
        Most Robust to Outliers: LMEDS and RANSAC are generally considered the most robust to outliers.
        Conditionally Robust: M-Estimators and IRLS can be made robust by carefully choosing loss functions and weighting schemes, respectively.
        Balanced Robustness and Efficiency: M-Estimators offer a good balance if the appropriate loss function is chosen.
        Somewhat Robust: Theil-Sen can be affected due to random selection of samples, if samples contain major outliers.
    """


    # Generate synthetic data
    np.random.seed(0)
    normal = np.array([0, 0, 1])
    d = 0
    points = np.random.randn(NUMBER_OF_POINTS, 3)
    points[:, :2] *= 0.1
    points[:, 2] = points[:, 0] * normal[0] + points[:, 1] * normal[1] + d

    # Add noise and outliers
    points[:NUMBER_OF_OUTLIERS] += np.random.randn(NUMBER_OF_OUTLIERS, 3) * 0.2
    outliers = np.random.rand(10, 3) * 2 - 1
    points_with_outliers = np.vstack([points, outliers])

    # Run RANSAC
    (normal_ransac, d_ransac), best_inliers = ransac(
        points_with_outliers, 
        threshold=THRESHOLD_RANSAC, 
        iterations=NUMBER_OF_ITERATIONS_RANSAC
    )
    plot_results(
        points_with_outliers=points_with_outliers, 
        a=normal_ransac[0], 
        b=normal_ransac[1], 
        c=normal_ransac[2], 
        d=d_ransac, 
        title="RANSAC"
    )

    # Run LMedS
    (normal_lmeds, d_lmeds), best_inliers_lmeds = least_median_squares(
        points_with_outliers, 
        iterations=NUMBER_OF_ITERATIONS_LMEDS
    )
    plot_results(
        points_with_outliers=points_with_outliers, 
        a=normal_lmeds[0], 
        b=normal_lmeds[1], 
        c=normal_lmeds[2], 
        d=d_lmeds, 
        title="LMedS"
    )

    # Fit plane using M-Estimator
    (normal_m_estimator, d_m_estimator) = fit_plane_m_estimator(
        points, 
        iterations=NUMBER_OF_ITERATIONS_M_ESTIMATOR,
        threshold=THRESHOLD_M_ESTIMATOR,
        alpha=ALPHA_M_ESTIMATOR
    )
    plot_results(
        points_with_outliers=points_with_outliers, 
        a=normal_m_estimator[0], 
        b=normal_m_estimator[1], 
        c=normal_m_estimator[2], 
        d=d_m_estimator, 
        title="M-Estimator"
    )

    # Run IRLS
    (normal_irls, d_irls) = fit_plane_IRLS(
        points, 
        iterations=NUMBER_OF_ITERATIONS_IRLS,
        epsilon=EPSILON_IRLS
    )
    plot_results(
        points_with_outliers=points_with_outliers, 
        a=normal_irls[0], 
        b=normal_irls[1],
        c=normal_irls[2],
        d=d_irls,
        title="IRLS"
    )

    # Run Theil-Sen Estimator
    (normal_theil_sen, d_theil_sen) = theil_sen_estimator(
        points=points,
        iterations=NUMBER_OF_ITERATIONS_THEIL_SEN
    )
    plot_results(
        points_with_outliers=points_with_outliers, 
        a=normal_theil_sen[0], 
        b=normal_theil_sen[1],
        c=normal_theil_sen[2],
        d=d_theil_sen,
        title="Theil-Sen"
    )
