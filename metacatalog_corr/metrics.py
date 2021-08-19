import numpy as np
from scipy import stats
import dcor
import ennemi
import minepy


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

def mutual_information_corr(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Mutual Information estimation between left and right array.
    Argument 'normalize=True' in **kwargs normalizes result to correlation coefficient scale.
    """
    corr = ennemi.estimate_mi(left, right, **kwargs)[0][0]

    return corr

def maximal_information_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Maximal information coefficient (MIC) for left and right array.
    """
    mine = minepy.MINE(alpha=0.6, c=15)
    mine.compute_score(left, right)
    corr = mine.mic()

    return corr
