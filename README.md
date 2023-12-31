# Robust-Plane-Fit
Various methods for fitting planes in the presence of outliers

## Robust to Non-Gaussian Noise and Deterministic

Extremely fast: M-Estimator (Early stopping might not happen in all cases, which can degrade below IRLS performance)

Relatively fast: Iteratively Reweighted Least Squares (More deterministic than M-Estimator)

## Robust to Gaussian Noise, Non-Deterministic

Medium with early termination: RANSAC

Slow: Least Median of Squares (more robust than RANSAC in the presence of noise)

## Can Handle A Small Proportion of Gaussian ONLY Noise

Extremely fast: Theil-Sen Estimator