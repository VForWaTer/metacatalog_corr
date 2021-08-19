import numpy as np
from scipy import stats
import dcor
import ennemi
import minepy
import hyppo.independence


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
    
    # MI becomes -Inf sometimes (if left == right?), which can´t be inserted into table correlation_matrix
    if np.isinf(corr):
        corr = np.nan

    return corr

def maximal_information_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Maximal information coefficient (MIC) for left and right array.
    """
    mine = minepy.MINE(alpha=0.6, c=15)
    mine.compute_score(left, right)
    corr = mine.mic()

    return corr

def kendall_tau_corr(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of Kendall's tau for left and right array.
    """
    corr, _ = stats.kendalltau(left, right, **kwargs)

    return corr

def weighted_tau_corr(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of a weighted version of Kendall's tau for left and right array.
    """
    corr, _ = stats.weightedtau(left, right)

    return corr

def somers_d_corr(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of Somers´ D for left and right array.
    """
    corr = stats.somersd(left, right)
    corr = corr.statistic

    return corr

#def heller_heller_gorfine(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
#    """
#    Calculate Heller-Heller-Gorfine test statistic for left and right array.
#    """
#    hhg = independence.HHG()
#    corr, _ = hhg.test(left,right)
#
#    return corr

