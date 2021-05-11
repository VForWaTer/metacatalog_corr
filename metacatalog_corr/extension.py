from sqlalchemy.orm import object_session
from sqlalchemy import func

from metacatalog import __version__
from metacatalog.ext import MetacatalogExtensionInterface
from metacatalog.models import Entry
from tqdm import tqdm

from metacatalog_corr import models


def find_correlated_data(self, limit: int = 50, metric: str = 'pearson', identifier: str = None, symmetric=True, return_iterator = False, **kwargs):
    """
    Find other Entry instances with correlating data.
    It is recommended to limit the result, as this function can potentially run
    for a long time returning huge amounts of data.
    If additional keyword arguments are given, these will be passed to 
    :func:`find_entry <metacatalog.api.find_entry>` to pre-filter the results.

    Parameters
    ----------
    limit : int
        Limit the results. The results will be ordered by correlation value.
        It is highly recommended to place a limit to the results.
        Defaults to 50
    metric : str
        Has to be a valid :class:`CorrelationMetric <metacatalog_corr.models.CorrelationMetric>`.
        Only matrix values for this metrix will be returned. Defaults to pearson.
    identifier : str
        If different matrix versions were indexed using an identifier, this can be used
        to load the correct matrix. This parameter filters on exact matches.
    symmetric : bool
        If True (default), the absolute correlation values will be used as
        negative correlation are assumed to be as strong as positive correlations.
        If set to False and limit is not None, the negative values will be ordered
        **after** the positives.
    return_iterator : bool
        If True, the function will not execute the query but return the query object.

    """
    from metacatalog import api

    # get the session
    session = object_session(self)

    # build the query - search for my entries
    query = session.query(models.CorrelationMatrix).filter(models.CorrelationMatrix.left_id==self.id)

    # filter for metric
    if metric is not None:
        query = query.join(models.CorrelationMetric).filter(models.CorrelationMetric.symbol==metric)

    # apply filtering
    if len(kwargs.keys()) > 0:
        # get other Entry ids
        others = [e.id for e in api.find_entry(session, **kwargs) if e.id != self.id]
        if len(others) > 0:
            query = query.filter(models.CorrelationMatrix.right_id.in_(others))

    # apply order
    if symmetric:
        query = query.order_by(func.abs(models.CorrelationMatrix.value).desc())
    else:
        query = query.order_by(models.CorrelationMatrix.value.desc())
    
    # apply limit
    if limit is not None:
        query = query.limit(limit)

    # handle output
    if return_iterator:
        return query
    else:
        return query.all()


def index_correlation_matrix(self: Entry, others: list, metrics = ['pearson'], if_exists='omit', commit=True, verbose=False, **kwargs):
    """
    .. note::
        This function is part of the ``metacatalog_corr`` extension.
        After installation, you need to enable the extension to use this function.
    
    Index the correlation matrix for this Entry by calculating each 
    given metric with all other Entries.

    Parameter
    ---------
    other : list
        List of other entries. This can be a list of 
        :class:`Entries <metacatalog.models.Entry>`, int (Entry.id) or
        str (Entry.uuid)
    metrics : list
        List of metrics to calculate. Each string in the list has to be
        available as CorrelationMetric.symbol in the database.
        Defaults to ``'pearson'``.
    if_exists : str
        Can be one of ``['omit', 'replace']``. Defaults to ``'omit'``.
        If a matrix cell is already filled, it can either be omitted
        or replaced.
    commit : bool
        If True (default), the calculated values will be persisted into
        the database.
    verbose : bool
        Enable text output.
    
    Returns
    -------
    correlation_matrix : List[CorrelationMatrix]
        List of CorrelationMatrix values calcualted. If existing 
        cells are omitted, they will **not** be in the list.

    """
    # pre-load the data
    left_df = self.get_data()
    left = left_df[self.datasource.column_names].values
    
    # handle verbosity
    if verbose:
        others = tqdm(others, unit='cells')
    
    # get a session
    session = object_session(self)

    # load the metrics
    metrics_objects = session.query(models.CorrelationMetric).filter(models.CorrelationMetric.symbol.in_(metrics)).all()

    output = []

    # go 
    for other in others:
        for metric in metrics_objects:
            # calculate
            cell = models.CorrelationMatrix.create(
                session,
                self,
                other,
                metric,
                commit=commit,
                start=kwargs.get('start'),
                end=kwargs.end('end'),
                identifier=kwargs.get('identifier'),
                left_data=left,
                if_exists=if_exists
            )
            # append
            output.append(cell)

    return output


class CorrExtension(MetacatalogExtensionInterface):
    """
    Correlation Extenstion.

    This extension will introduce two new tables to metacatalog.
    A lookup table for correlation metrics, with some pre-defined
    values. Secondly, a correlation matrix table, that relates
    two metacatalog.Entry records via foreign keys. 
    The extension has a submodule called metrics that contains
    the actual Python implementations for the metrics, but the
    extension is in principle capable of utilizing any Python
    package, as long as the signature is correct.

    .. code-block::
        def correlation_func(left: np.ndarray, right: np.ndarray, **kwargs) -> float:
            pass

    The metacatalog.Entry receives two new instance methods:

    * index_correlation_matrix
    * find_correlated_data

    With index_correlation_matrix, the Entry will fill the 
    correlation matrix with specified metrics for all given
    other Entry records. This method can be added to the load
    event of Entry, which is not done by default. This function
    can potentially take forever, depending on the data amount stored
    in the database and it may be a wise decision to call it 
    on a regular basis.
    With find_correlated_data, the Entry can list other Entries
    with potentially similar data. For this to work, the correlation
    matrix has to be indexed first.

    """
    @classmethod
    def check_version(cls):
        major, minor, patch = __version__.split('.')

        return int(major) > 0 or int(minor) >= 3

    @classmethod
    def init_extension(cls):
        """
        For initializing the extension several step are needed.
        First merge the declarative base and add missing foreign
        keys column declarations.
        """
        if not cls.check_version:
            raise RuntimeError(f"[metacatalog_corr] Metacatalog is too old. Version >= 0.3.0 needed, found: {__version__}")

        # merge the declarative base
        from metacatalog.db.base import Base
        models.merge_declarative_base(Base.metadata)

        # add new methods
        Entry.find_correlated_data = find_correlated_data
        Entry.index_correlation_matrix = index_correlation_matrix
