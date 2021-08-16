import numpy as np
from scipy import stats
import dcor


def pearson_corr_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Pearson correlation coefficient for left and right array.

    """
    corr, _ = stats.pearsonr(left, right)
    
    return corr


def spearman_corr_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Spearman rank correlation coeficient for left and right array.
    """
    corr, _ = stats.spearmanr(left, right, **kwargs)

    return corr

def distance_corr(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Distance correlation for left and right array.
    """
    corr = dcor.distance_correlation(left, right, **kwargs)

    return corr
