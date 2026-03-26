import numpy as np
from soak_split import soak_split


def test_split_runs():
    subset_vec = np.array([0, 0, 1, 1, 2, 2])
    result = soak_split(subset_vec, n_splits=3, seed=42)

    assert isinstance(result, list)
    assert len(result) > 0


def test_output_structure():
    subset_vec = np.array([0, 1, 0, 1])
    result = soak_split(subset_vec, n_splits=2)

    item = result[0]

    # Expect 5 elements per entry
    assert len(item) == 5

    test_subset, category, fold_id, train_idx, test_idx = item

    assert category in {"same", "other", "all"}
    assert isinstance(fold_id, int)
    assert fold_id >= 1


def test_indices_are_valid():
    subset_vec = np.array([0, 0, 1, 1])
    result = soak_split(subset_vec, n_splits=2)

    n = len(subset_vec)

    for _, _, _, train_idx, test_idx in result:
        assert np.all(train_idx < n)
        assert np.all(test_idx < n)


def test_no_overlap_between_train_and_test():
    subset_vec = np.array([0, 0, 1, 1])
    result = soak_split(subset_vec, n_splits=2)

    for _, _, _, train_idx, test_idx in result:
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_all_categories_present():
    subset_vec = np.array([0, 0, 1, 1])
    result = soak_split(subset_vec, n_splits=2)

    categories = {item[1] for item in result}

    assert {"same", "other", "all"}.issubset(categories)