# utils/statistics.py
# From winter_rl_10seed_final_v3.py (unchanged)

import numpy as np
from scipy import stats

def mean_std_ci(arr, confidence=0.95):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    m = arr.mean()
    s = arr.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        se = s / np.sqrt(n)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        ci = t_crit * se
    else:
        ci = 0.0
    return m, s, m - ci, m + ci

def welch_t_test(a, b):
    return stats.ttest_ind(a, b, equal_var=False)
