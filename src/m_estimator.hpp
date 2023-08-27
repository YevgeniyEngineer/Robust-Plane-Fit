#ifndef M_ESTIMATOR_HPP
#define M_ESTIMATOR_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/src/Core/util/Constants.h>
#include <cstdint>

namespace plane_fit
{
template <typename FloatType>
std::pair<Eigen::Matrix<FloatType, 3, 1>, FloatType> fitPlaneWithMEstimator(
    const Eigen::Matrix<FloatType, Eigen::Dynamic, 3> &points, std::uint32_t iterations = 100U,
    FloatType threshold = static_cast<FloatType>(1.0), FloatType alpha = static_cast<FloatType>(0.01),
    FloatType convergence_tolerance = static_cast<FloatType>(1e-6), FloatType epsilon = static_cast<FloatType>(1e-6))
{
    std::uint32_t num_points = points.rows();

    // Initialize the plane parameters (a, b, c, d) for the equation ax + by + cz = d
    Eigen::Matrix<FloatType, 4, 1> plane_coefficients;
    Eigen::Matrix<FloatType, 4, 1> old_plane_coefficients;

    plane_coefficients << FloatType(0.0), FloatType(0.0), FloatType(1.0), FloatType(0.0);
    old_plane_coefficients = plane_coefficients;

    Eigen::Matrix<FloatType, Eigen::Dynamic, 1> residuals(num_points);
    Eigen::Matrix<bool, Eigen::Dynamic, 1> mask(num_points);
    Eigen::Matrix<FloatType, Eigen::Dynamic, 4> A(num_points, 4);
    Eigen::DiagonalMatrix<FloatType, Eigen::Dynamic> W(num_points);
    Eigen::Matrix<FloatType, 4, 4> ATW;

    // Prepare design matrix A
    A.block(0, 0, num_points, 3) = points;
    A.col(3) = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>::Ones(num_points);

    // Preallocate JacobiSVD
    Eigen::JacobiSVD<Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>> svd(
        4, 4, Eigen::ComputeThinU | Eigen::ComputeThinV);

    for (std::uint32_t i = 0; i < iterations; ++i)
    {
        // Calculate the residuals
        residuals.array() = plane_coefficients[0] * points.col(0).array() +
                            plane_coefficients[1] * points.col(1).array() +
                            plane_coefficients[2] * points.col(2).array() - plane_coefficients[3];

        // Update the mask based on the residuals
        mask = (residuals.array().abs() > threshold).template cast<bool>();

        // Calculate Huber weights for each point
        W.diagonal().array() = mask.select((threshold / (residuals.array().abs() + epsilon)), 1);

        // Update the parameters using SVD
        ATW.noalias() = A.transpose() * W;
        svd.compute(ATW * A);
        plane_coefficients = svd.solve(ATW * points.col(2));

        // Check for convergence
        if ((plane_coefficients - old_plane_coefficients).norm() < convergence_tolerance)
        {
            break;
        }

        old_plane_coefficients = plane_coefficients;
    }

    const Eigen::Matrix<FloatType, 3, 1> normal_vector = plane_coefficients.template head<3>();
    const FloatType d = plane_coefficients[3];

    return std::make_pair(normal_vector, d);
}
} // namespace plane_fit

#endif // M_ESTIMATOR_HPP