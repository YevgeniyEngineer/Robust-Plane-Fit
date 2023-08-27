#ifndef RANDOM_SAMPLE_CONSENSUS_HPP
#define RANDOM_SAMPLE_CONSENSUS_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace plane_fit
{
template <typename FloatType>
std::pair<Eigen::Matrix<FloatType, 3, 1>, FloatType> fitPlaneWithRandomSampleConsensus(
    const Eigen::Matrix<FloatType, Eigen::Dynamic, 3> &points, FloatType threshold, std::uint32_t iterations,
    FloatType p = static_cast<FloatType>(0.999))
{
    Eigen::Matrix<FloatType, 3, 1> best_normal;
    FloatType best_d = static_cast<FloatType>(0.0);
    std::uint32_t max_inliers = 0;

    // Initial maximum number of iterations
    std::uint32_t N = iterations;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<std::int32_t> distribution(0, points.rows() - 1);

    Eigen::Matrix<FloatType, Eigen::Dynamic, 1> residuals(points.rows());
    Eigen::Matrix<FloatType, 3, 3> sample_points;
    Eigen::Matrix<FloatType, 3, 1> normal_vector;
    FloatType d = static_cast<FloatType>(0.0);

    std::uint32_t iteration = 0;
    while (iteration < N)
    {
        // Generate three unique random indices
        const auto idx1 = distribution(rng);
        auto idx2 = distribution(rng);
        while (idx2 == idx1)
        {
            idx2 = distribution(rng);
        }
        auto idx3 = distribution(rng);
        while ((idx3 == idx1) || (idx3 == idx2))
        {
            idx3 = distribution(rng);
        }

        sample_points.row(0) = points.row(idx1);
        sample_points.row(1) = points.row(idx2);
        sample_points.row(2) = points.row(idx3);

        // Fit a plane to the sample points
        normal_vector =
            (sample_points.row(1) - sample_points.row(0)).cross(sample_points.row(2) - sample_points.row(0));

        // Normalize the normal vector
        normal_vector.normalize();
        d = sample_points.row(0).dot(normal_vector);

        residuals = ((points * normal_vector).array() - d).abs();
        auto num_inliers = (residuals.array() < threshold).count();

        if (num_inliers > max_inliers)
        {
            max_inliers = num_inliers;
            best_normal = normal_vector;
            best_d = d;

            const FloatType e = 1.0 - static_cast<FloatType>(num_inliers) / points.rows();
            if (e < 1.0)
            {
                N = static_cast<std::uint32_t>(std::log(1.0 - p) / std::log(1.0 - std::pow(1.0 - e, 3.0)));
            }
        }

        ++iteration;
    }

    return std::make_pair(best_normal, best_d);
}
} // namespace plane_fit

#endif // RANDOM_SAMPLE_CONSENSUS_HPP