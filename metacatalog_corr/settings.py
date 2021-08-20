DEFAULT_METRICS = [
    {
        'symbol': 'pearson', 
        'name': 'Pearson correlation coefficient',
        'description': 'Pearson correlation coefficient, as defined in https://en.wikipedia.org/wiki/Pearson_correlation_coefficient; Implementation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html',
        'function_name': 'pearson_corr_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'spearman',
        'name': 'Spearman rank correlation test',
        'description': 'Non-parametric correlation test; Implementation from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html',
        'function_name': 'spearman_corr_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'nan_policy': 'omit'}
    },
    {
        'symbol': 'dcor',
        'name': 'Distance correlation',
        'description': 'Calculation of the usual (biased) estimator for the distance correlation; Implementation from https://dcor.readthedocs.io/en/latest/functions/dcor.distance_correlation.html#dcor.distance_correlation',
        'function_name': 'distance_corr',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'exponent': 1}
    },
    {
        'symbol': 'mutual_info',
        'name': 'Mutual Information',
        'description': 'Estimation of the mutual information; Implementation from https://polsys.github.io/ennemi/api-reference.html#:~:text=to%20discrete%20data.-,estimate_mi,-Estimate%20the%20mutual',
        'function_name': 'mutual_information_corr',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'normalize': True}
    },
    {
        'symbol': 'mic',
        'name': 'Maximal information coefficient',
        'description': 'Calulation of the maximal information coefficient; Implementation from https://minepy.readthedocs.io/en/latest/python.html',
        'function_name': 'maximal_information_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'kendall_tau',
        'name': 'Kendall´s tau',
        'description': 'Calulation of Kendall´s tau correlation measure; Implementation from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html#scipy.stats.kendalltau',
        'function_name': 'kendall_tau_corr',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'nan_policy': 'omit'}
    },
    {
        'symbol': 'weighted_tau',
        'name': 'Weighted Kendall´s tau',
        'description': 'Calulation of weighted Kendall´s tau correlation measure; Implementation from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html#scipy.stats.weightedtau',
        'function_name': 'weighted_tau_corr',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'somers_d',
        'name': 'Somers´ D',
        'description': 'Calculation of Somers´ D correlation measure; Implementation from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.somersd.html#scipy.stats.somersd',
        'function_name': 'somers_d_corr',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'hoeffdings_d',
        'name': 'Hoeffding´s D',
        'description': 'Calculation of Hoeffding´s D correlation measure; Implementation from https://github.com/PaulVanDev/HoeffdingD',
        'function_name': 'hoeffdings_d_coeff',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'biweight_midcorr',
        'name': 'Biweight midcorellation',
        'description': 'Calculation of the biweight midcorrelation; Implementation from https://docs.astropy.org/en/stable/api/astropy.stats.biweight_midcorrelation.html',
        'function_name': 'biweight_midcorr_coeff',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    }
]