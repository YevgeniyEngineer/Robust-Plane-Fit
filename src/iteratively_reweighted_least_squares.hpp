#ifndef ITERATIVELY_REWEIGHTED_LEAST_SQUARES_HPP
#define ITERATIVELY_REWEIGHTED_LEAST_SQUARES_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cstdint>
#include <utility>

namespace plane_fit
{
template <typename FloatType>
std::pair<Eigen::Matrix<FloatType, 3, 1>, FloatType> fitPlaneWithIterativelyReweightedLeastSquares(
    const Eigen::Matrix<FloatType, Eigen::Dynamic, 3> &points, std::uint32_t iterations = 10,
    FloatType epsilon = static_cast<FloatType>(1e-6))
{
    using VectorType = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;
    using DiagonalMatrixType = Eigen::DiagonalMatrix<FloatType, Eigen::Dynamic>;

    // Prepare the design matrix X
    MatrixType X(points.rows(), 4);
    X.block(0, 0, points.rows(), 3) = points;   // Copy XYZ coordinates
    X.col(3) = VectorType::Ones(points.rows()); // Add ones to last column

    // Initialize coefficients and weight matrix
    VectorType plane_coefficients(4);
    plane_coefficients.setZero();

    DiagonalMatrixType W(VectorType::Ones(points.rows()));

    // Preallocate memory
    MatrixType XTW(X.cols(), points.rows());

    // Preallocate SVD object
    Eigen::JacobiSVD<MatrixType> svd(X.rows(), X.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV);

    for (std::uint32_t i = 0; i < iterations; ++i)
    {
        // Construct weighted design matrix and target vector
        XTW.noalias() = X.transpose() * W;

        // Weighted least squares solution using SVD
        svd.compute(XTW * X);

        // Solve for YW = XTW * points.col(2)
        plane_coefficients = svd.solve(XTW * points.col(2));

        // Update weights
        W.diagonal().array() = 1.0 / ((points.col(2) - X * plane_coefficients).cwiseAbs().array() + epsilon);
    }

    const Eigen::Matrix<FloatType, 3, 1> normal_vector = plane_coefficients.template head<3>();
    const FloatType d = plane_coefficients(3);

    return std::make_pair(normal_vector, d);
}

} // namespace plane_fit

#endif // ITERATIVELY_REWEIGHTED_LEAST_SQUARES_HPP