import numpy as np


def compute_errors(gt, pred, max_depth=np.inf):
    valid1 = gt >= 0.5
    valid2 = gt <= max_depth
    valid = valid1 & valid2
    gt = gt[valid]
    pred = pred[valid]

    n_valid = np.float32(len(gt))
    if n_valid == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    differences = gt - pred
    abs_differences = np.abs(differences)
    squared_differences = np.square(differences)
    abs_error = np.mean(abs_differences)
    abs_relative_error = np.mean(abs_differences / gt)
    abs_inverse_error = np.mean(np.abs(1 / gt - 1 / pred))
    squared_relative_error = np.mean(squared_differences / gt)
    rmse = np.sqrt(np.mean(squared_differences))
    ratios = np.maximum(gt / pred, pred / gt)
    n_valid = np.float32(len(ratios))
    ratio_125 = np.count_nonzero(ratios < 1.25) / n_valid
    ratio_125_2 = np.count_nonzero(ratios < 1.25 ** 2) / n_valid
    ratio_125_3 = np.count_nonzero(ratios < 1.25 ** 3) / n_valid
    return abs_error, abs_relative_error, abs_inverse_error, squared_relative_error, rmse, ratio_125, ratio_125_2, ratio_125_3


def sanity_check_compute_errors():
    gt = np.ones((256, 256), dtype=float) / 2.0
    pred = gt + np.random.normal(loc=0.0, scale=0.1, size=(np.shape(gt)))
    print(compute_errors(gt, pred))


if __name__ == '__main__':
    sanity_check_compute_errors()
