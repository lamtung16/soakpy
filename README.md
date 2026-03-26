# SOAK: Same/Other/All K-fold Cross-Validation
SOAK is designed to estimate the **similarity of patterns** found across different subsets of a dataset. It extends traditional K-fold cross-validation with "Same," "Other," and "All" splitting strategies to provide a robust measure of pattern similarity.

# Usage
```python
import numpy as np
import soak_split

# --- synthetic data ---
X = np.arange(8).reshape(-1, 1)
y = X.ravel()
subset_vec = np.array(['even' if x % 2 == 0 else 'odd' for x in X.ravel()])

# --- Initialize soak object ---
for subset_value, category, fold_id, train_idx, test_idx in soak_split.soak_split(subset_vec, n_splits=2):
    print(f"subset: {subset_value:6s} --- category: {category:6s} --- fold: {fold_id}")
    print(f"y_test:  {y[test_idx]}")
    print(f"y_train: {y[train_idx]}")
    print("-"*50)
```

```
subset: even   --- category: same   --- fold: 1
y_test:  [0]
y_train: [2 4 6]
--------------------------------------------------
subset: even   --- category: other  --- fold: 1
y_test:  [0]
y_train: [5]
--------------------------------------------------
subset: even   --- category: all    --- fold: 1
y_test:  [0]
y_train: [2 4 5 6]
--------------------------------------------------
subset: odd    --- category: same   --- fold: 1
y_test:  [1 3 7]
y_train: [5]
--------------------------------------------------
subset: odd    --- category: other  --- fold: 1
y_test:  [1 3 7]
y_train: [2 4 6]
--------------------------------------------------
subset: odd    --- category: all    --- fold: 1
y_test:  [1 3 7]
y_train: [2 4 5 6]
--------------------------------------------------
subset: even   --- category: same   --- fold: 2
y_test:  [2 4 6]
y_train: [0]
--------------------------------------------------
subset: even   --- category: other  --- fold: 2
y_test:  [2 4 6]
y_train: [1 3 7]
--------------------------------------------------
subset: even   --- category: all    --- fold: 2
y_test:  [2 4 6]
y_train: [0 1 3 7]
--------------------------------------------------
subset: odd    --- category: same   --- fold: 2
y_test:  [5]
y_train: [1 3 7]
--------------------------------------------------
subset: odd    --- category: other  --- fold: 2
y_test:  [5]
y_train: [0]
--------------------------------------------------
subset: odd    --- category: all    --- fold: 2
y_test:  [5]
y_train: [0 1 3 7]
--------------------------------------------------
```