#ifndef LEAST_MEDIAN_OF_SQUARES_HPP
#define LEAST_MEDIAN_OF_SQUARES_HPP

#include <Eigen/Core>
#include <cstdint>
#include <random>

namespace plane_fit
{
template <typename FloatType>
std::pair<Eigen::Matrix<FloatType, 3, 1>, FloatType> fitPlaneWithLeastMedianOfSquares(
    const Eigen::Matrix<FloatType, Eigen::Dynamic, 3> &points, std::uint32_t iterations, FloatType p = 0.999)
{
    Eigen::Matrix<FloatType, 3, 1> best_normal;
    FloatType best_d = 0.0;
    FloatType min_median = std::numeric_limits<FloatType>::max();

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<std::int32_t> distribution(0, points.rows() - 1);

    Eigen::Matrix<FloatType, Eigen::Dynamic, 1> residuals(points.rows());
    Eigen::Matrix<FloatType, 3, 3> sample_points;
    Eigen::Matrix<FloatType, 3, 1> normal_vector;
    FloatType d;

    std::uint32_t N = iterations;
    std::uint32_t iteration = 0;

    while (iteration < N)
    {
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

        // Calculate normal vector
        normal_vector =
            (sample_points.row(1) - sample_points.row(0)).cross(sample_points.row(2) - sample_points.row(0));
        normal_vector.normalize();
        d = sample_points.row(0).dot(normal_vector);

        residuals = ((points * normal_vector).array() - d).abs();

        // Approximate median
        std::nth_element(residuals.data(), residuals.data() + residuals.size() / 2,
                         residuals.data() + residuals.size());

        // Check convergence and update best candidate
        const FloatType median_error = residuals(residuals.size() / 2);
        if (median_error < min_median)
        {
            min_median = median_error;
            best_normal = normal_vector;
            best_d = d;

            const FloatType w = 0.5;
            N = static_cast<std::uint32_t>(std::log(1.0 - p) / std::log(1.0 - std::pow(w, 3.0)));
        }

        ++iteration;
    }

    return std::make_pair(best_normal, best_d);
}
} // namespace plane_fit

#endif // LEAST_MEDIAN_OF_SQUARES_HPP