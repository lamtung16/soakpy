# src/soakpy/core.py

import numpy as np
from sklearn.model_selection import KFold


def split(subset_vec, n_splits=5, seed=0):
    """
    Perform SOAK splitting.

    Parameters
    ----------
    subset_vec : array-like
        Vector indicating subset/group membership.
    n_splits : int, default=5
        Number of folds.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    list
        List of [test_subset, category, fold_id, train_idx, test_idx]
    """
    
    subset_vec = np.asarray(subset_vec)

    if subset_vec.ndim != 1:
        raise ValueError("subset_vec must be a 1D array-like")

    n = subset_vec.shape[0]

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if n_splits > n:
        raise ValueError("n_splits cannot exceed number of samples")
    
    splits = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx = np.arange(n)

    for fold_id, (fold_train_idx, fold_test_idx) in enumerate(kf.split(range(n))):
        for test_subset in np.unique(subset_vec):
            test_subset_idx = np.where(subset_vec == test_subset)[0]
            test_idx = np.intersect1d(test_subset_idx, fold_test_idx)

            train_idx_dict = {
                "same":  test_subset_idx,
                "other": np.setdiff1d(all_idx, test_subset_idx),
                "all":   all_idx,
            }

            for category, train_idx in train_idx_dict.items():
                splits.append((test_subset, category, fold_id + 1, np.intersect1d(fold_train_idx, train_idx), test_idx))

    return splits