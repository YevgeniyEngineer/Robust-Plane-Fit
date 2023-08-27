#include "iteratively_reweighted_least_squares.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <new>
#include <random>

template <typename FloatType>
Eigen::Matrix<FloatType, Eigen::Dynamic, 3> generateNoisyPlanePoints(std::int32_t num_points, FloatType noise_level,
                                                                     FloatType sine_amplitude, FloatType sine_frequency)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<FloatType> dis(FloatType(-10.0),
                                                  FloatType(10.0)); // Uniform distribution between -10 and 10
    std::normal_distribution<FloatType> noise_dis(FloatType(0.0), noise_level); // Normal distribution for noise

    Eigen::Matrix<FloatType, Eigen::Dynamic, 3> points(num_points, 3);
    for (std::int32_t i = 0; i < num_points; ++i)
    {
        const FloatType x = dis(gen);
        const FloatType y = dis(gen);

        // Add some noise to the z-coordinate
        // Points lie on the plane z=0
        FloatType z = FloatType(0.0);

        // Add Gaussian noise to the z-coordinate
        z += noise_dis(gen);

        // Add sine bias to the z-coordinate
        z += sine_amplitude * std::sin(sine_frequency * x);

        points.row(i) << x, y, z;
    }

    return points;
}

int main()
{
    constexpr std::int32_t NUMBER_OF_POINTS = 120'000;
    constexpr std::uint32_t NUMBER_OF_ITERATIONS = 5;
    constexpr std::uint32_t TIMING_ITERATIONS = 100;

    using FloatType = float;

    try
    {
        // Generate 100 random points on a horizontal plane with a noise level of 1.0,
        // sine amplitude of 2.0, and sine frequency of 0.5
        const Eigen::Matrix<FloatType, Eigen::Dynamic, 3> points =
            generateNoisyPlanePoints<FloatType>(NUMBER_OF_POINTS, 1.0, 6.0, 0.5);

        for (auto timing_iteration = 0; timing_iteration < TIMING_ITERATIONS; ++timing_iteration)
        {
            auto t1 = std::chrono::high_resolution_clock::now();

            // Fit a plane to the points
            auto result =
                plane_fit::fitPlaneWithIterativelyReweightedLeastSquares<FloatType>(points, NUMBER_OF_ITERATIONS);

            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << "Plane coefficients (a, b, c, d): " << result.first.transpose() << " " << result.second
                      << std::endl;
            std::cout << "Elapsed time [microseconds]: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
        }
    }
    catch (const std::bad_alloc &ex)
    {
        std::cerr << "Could not allocate memory. Exception: " << ex.what() << std::endl;
    }

    return 0;
}