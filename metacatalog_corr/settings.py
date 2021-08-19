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
    }
]