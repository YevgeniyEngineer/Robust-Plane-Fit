#ifndef THEIL_SEN_ESTIMATOR_HPP
#define THEIL_SEN_ESTIMATOR_HPP

#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace plane_fit
{
template <typename FloatType>
std::pair<Eigen::Matrix<FloatType, 3, 1>, FloatType> fitPlaneWithTheilSenEstimator(
    const Eigen::Matrix<FloatType, Eigen::Dynamic, 3> &points, std::uint32_t iterations)
{
    std::vector<Eigen::Matrix<FloatType, 3, 1>> normals;
    std::vector<FloatType> ds;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<std::int32_t> distribution(0, points.rows() - 1);

    Eigen::Matrix<FloatType, 3, 3> sample_points;
    Eigen::Matrix<FloatType, 3, 1> normal_vector;
    FloatType d;

    for (std::uint32_t i = 0; i < iterations; ++i)
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

        normal_vector =
            (sample_points.row(1) - sample_points.row(0)).cross(sample_points.row(2) - sample_points.row(0));
        normal_vector.normalize();
        d = sample_points.row(0).dot(normal_vector);

        normals.push_back(normal_vector);
        ds.push_back(d);
    }

    Eigen::Matrix<FloatType, Eigen::Dynamic, 3> all_normals(iterations, 3);
    for (std::uint32_t i = 0; i < iterations; ++i)
    {
        all_normals.row(i) = normals[i];
    }

    Eigen::Matrix<FloatType, Eigen::Dynamic, 1> all_ds(iterations);
    for (std::uint32_t i = 0; i < iterations; ++i)
    {
        all_ds(i) = ds[i];
    }

    // Calculate the median of normals and d values
    Eigen::Matrix<FloatType, 3, 1> median_normal;
    for (std::int32_t i = 0; i < 3; ++i)
    {
        std::nth_element(all_normals.col(i).data(), all_normals.col(i).data() + all_normals.rows() / 2,
                         all_normals.col(i).data() + all_normals.rows());
        median_normal(i) = all_normals(all_normals.rows() / 2, i);
    }

    std::nth_element(all_ds.data(), all_ds.data() + all_ds.size() / 2, all_ds.data() + all_ds.size());
    FloatType median_d = all_ds(all_ds.size() / 2);

    return std::make_pair(median_normal, median_d);
}
} // namespace plane_fit

#endif // THEIL_SEN_ESTIMATOR_HPP