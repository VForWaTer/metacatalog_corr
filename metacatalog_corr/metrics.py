import numpy as np
from scipy import stats
import dcor
import minepy
import pingouin
import skinfo.metrics as skinfo

#import hyppo.independence


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

def distance_corr_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Distance correlation for left and right array.
    """
    corr = dcor.distance_correlation(left, right, **kwargs)

    return corr

def maximal_information_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Maximal information coefficient (MIC) for left and right array.
    """
    # choose alpha depending on sample size n (speeds up mic calculation), Albanese et al. 2018, Tab. 1
    n = len(left)
    conditions = [
        n < 25, 
        25 <= n < 50, 
        50 <= n < 250, 
        250 <= n < 500, 
        500 <= n < 1000, 
        1000 <= n < 2500,
        2500 <= n < 5000,
        5000 <= n < 10000,
        10000 <= n < 40000,
        n >= 40000 
    ]
    choices = [.85, .8, .75, .7, .65, .6, .55, .5, .45, .4]
    alpha = np.select(conditions, choices)

    mine = minepy.MINE(alpha=alpha, c=15)
    mine.compute_score(left, right)
    corr = mine.mic()

    return corr

def kendall_tau_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of Kendall's tau for left and right array.
    """
    corr, _ = stats.kendalltau(left, right, **kwargs)

    return corr

def weighted_tau_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of a weighted version of Kendall's tau for left and right array.
    """
    corr, _ = stats.weightedtau(left, right)

    return corr

def somers_d_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of Somers´ D for left and right array.
    """
    corr = stats.somersd(left, right)
    corr = corr.statistic

    return corr

def biweight_midcorr_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of biweight mid correlation for left and right array.
    """
    corr = pingouin.corr(left, right, method='bicor')
    corr = float(corr.r)

    return corr

def percentage_bend_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of Percentage bend correlation for left and right array.
    """
    corr = pingouin.corr(left, right, method='percbend')
    corr = float(corr.r)

    return corr

def shepherds_pi_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of Shepherd's pie correlation coefficient for left and right array.
    """
    corr = pingouin.corr(left, right, method='shepherd')
    corr = float(corr.r)

    return corr

def skipped_corr_coef(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of the Skipped correlation coefficient for left and right array.
    """
    corr = pingouin.corr(left, right, method='skipped')
    corr = float(corr.r)

    return corr

def conditional_entropy(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of the Conditional entropy for left and right array.
    Normalized conditional entropy equals 1 if and only if left and right 
    are independent. Therefore, 1 - cond_entropy measures the dependency 
    between left and right array and creates comparability with abs(pearson).
    """
    corr = 1 - skinfo.conditional_entropy(left, right, **kwargs)

    # return 0 if the absolute value of conditional entropy is close to 0
    if abs(corr) < 1e-4:
        corr = 0
    return corr

def mutual_information(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of the Mutual information for left and right array.   
    """
    corr = skinfo.mutual_information(left, right, **kwargs)

    return corr

def jensen_shannon(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of the Jensen-Shannon Divergence and Distance for left and right array.
    
    JSD measures divergence / distance. Therefore, 1 - JSD measures the similarity 
    between left and right array and creates comparability with abs(pearson).
    """
    corr = 1 - skinfo.jensen_shannon(left, right, **kwargs)

    return corr


# not normalized metrics:
def cross_entropy(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of the Cross entropy for left and right array.   
    """
    corr = skinfo.cross_entropy(left, right, **kwargs)

    return corr

def kullback_leibler(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
    """
    Calculation of the Kullback Leibler Divergence for left and right array.   
    """
    corr = skinfo.kullback_leibler(left, right, **kwargs)

    return corr