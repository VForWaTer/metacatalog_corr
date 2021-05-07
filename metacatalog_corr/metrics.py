import numpy as np
from scipy import stats


def pearson_corr_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Pearson correlation coefficient for left and right array.

    """
    corr, _ = stats.pearsonr(left, right)
    
    return corr


def spearman_corr_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Spearman rank correlation coeficient for left and right array
    """
    corr, _ = stats.spearmanr(left, right, **kwargs)

    return corr
