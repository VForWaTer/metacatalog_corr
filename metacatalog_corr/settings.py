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
        'function_name': 'distance_corr_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'exponent': 1}
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
        'function_name': 'kendall_tau_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'nan_policy': 'omit'}
    },
    {
        'symbol': 'weighted_tau',
        'name': 'Weighted Kendall´s tau',
        'description': 'Calulation of weighted Kendall´s tau correlation measure; Implementation from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html#scipy.stats.weightedtau',
        'function_name': 'weighted_tau_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'somers_d',
        'name': 'Somers´ D',
        'description': 'Calculation of Somers´ D correlation measure; Implementation from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.somersd.html#scipy.stats.somersd',
        'function_name': 'somers_d_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'hoeffdings_d',
        'name': 'Hoeffding´s D',
        'description': 'Calculation of Hoeffding´s D correlation measure; Implementation from https://github.com/PaulVanDev/HoeffdingD',
        'function_name': 'hoeffdings_d_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'biweight_mid',
        'name': 'Biweight midcorellation',
        'description': 'Calculation of the biweight midcorrelation coefficient; Implementation from https://pingouin-stats.org/generated/pingouin.corr.html',
        'function_name': 'biweight_midcorr_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'perc_bend',
        'name': 'Percentage bend correlation',
        'description': 'Calculation of the Percentage bend correlation coefficient; Implementation from https://pingouin-stats.org/generated/pingouin.corr.html',
        'function_name': 'percentage_bend_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'shepherd',
        'name': 'Shepherd´s pie correlation',
        'description': 'Calculation of the Shepherd´s pie correlation coefficient; Implementation from https://pingouin-stats.org/generated/pingouin.corr.html',
        'function_name': 'shepherds_pi_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'skipped',
        'name': 'Skipped correlation',
        'description': 'Calculation of the Skipped correlation coefficient; Implementation from https://pingouin-stats.org/generated/pingouin.corr.html',
        'function_name': 'skipped_corr_coef',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {}
    },
    {
        'symbol': 'conditional_entropy',
        'name': 'Conditional entropy',
        'description': 'Calculation of the Conditional entropy; Implementation from https://github.com/KIT-HYD/scikit-info',
        'function_name': 'conditional_entropy',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'bins': 'auto'}
    },
    {
        'symbol': 'mutual_info',
        'name': 'Mutual Information',
        'description': 'Calculation of the mutual information; Implementation from https://github.com/KIT-HYD/scikit-info',
        'function_name': 'mutual_information',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'bins': 'auto'}
    },
    {
        'symbol': 'cross_entropy',
        'name': 'Cross entropy',
        'description': 'Calculation of the Cross entropy; Implementation from https://github.com/KIT-HYD/scikit-info',
        'function_name': 'cross_entropy',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'bins': 'auto'}
    },
    {
        'symbol': 'kullback_leibler',
        'name': 'Kullback Leibler',
        'description': 'Calculation of the Kullback Leibler Divergence; Implementation from https://github.com/KIT-HYD/scikit-info',
        'function_name': 'kullback_leibler',
        'import_path': 'metacatalog_corr.metrics',
        'function_args': {'bins': 'auto'}
    }
]

DEFAULT_WARNINGS = [
    {
    'category': 'NoDataWarning',
    'message': 'No data available for left and/or right entry due to datasource, harmonization or nan remove.'
    },
    {'category': 'HarmonizationWarning',
    'message': 'Indices of left and right data have no matches, harmonization not possible.'
    }
]