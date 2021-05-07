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
    }
]