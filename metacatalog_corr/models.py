import importlib
from datetime import datetime as dt
from typing import Union
from unicodedata import category
import warnings
from warnings import WarningMessage

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
import numpy as np
from sqlalchemy.orm.session import object_session

from skinfo.metrics import get_2D_bins


# create a new declarative base only for this extension
Base = declarative_base()


class CorrelationMetric(Base):
    """
    Defines the type of correlation metric.

    Attributes
    ----------
    id : int
        Unique id of the metric
    symbol : str
        Short abbreviated name or symbol usually
        associated to the metric
    name : str
        Full name of the metric
    descrption : str
        Description to the metric. Add references
        whenever possible or be as precise as 
        necessary

    """
    __tablename__ = 'correlation_metrics'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    symbol = sa.Column(sa.String(20), nullable=False)
    name = sa.Column(sa.String(500), nullable=False)
    description = sa.Column(sa.Text, nullable=True)
    function_name = sa.Column(sa.String(500), nullable=False)
    import_path = sa.Column(sa.String(1000), nullable=False)
    function_args = sa.Column(MutableDict.as_mutable(JSONB), nullable=False, default={})
    created = sa.Column(sa.DateTime, default=dt.utcnow)
    updated = sa.Column(sa.DateTime, default=dt.utcnow, onupdate=dt.utcnow)

    @property
    def func(self):
        """
        Load the actual function and return.
        """
        mod = importlib.import_module(self.import_path)
        func = getattr(mod, self.function_name)
        return func
    
    def calc(self, left: np.ndarray, right: np.ndarray, **kwargs) -> float:
        """
        Calculate the metric for the given data.
        """
        return self.func(left, right, **self.function_args, **kwargs)

    def permutation_test(self, left: np.ndarray, right: np.ndarray, n_iter=100, seed=None, **kwargs) -> float:
        """
        Calculate non-parametric permutation test for the given data
        
        Marozzi, 2004: n_iter proposals
        
        """  
        # calculate "true" correlation value
        true_corr = self.calc(left, right, **kwargs)
        # Initialize list to store permuted correlation scores
        perm_corr = []
        # set random seed (reproducibility master thesis: 42)
        np.random.seed(seed)
        # Initialize permutation loop:
        for iter in range(0, n_iter):
            # Shuffle right array:
            perm_right = np.random.permutation(right)
            # Compute permuted correlations and store them in perm_corr:
            perm_corr.append(self.calc(left, perm_right, **kwargs))
        # Significance: share of perm_corr which are >= true_corr (two-sided: absolute value):
        perm_p = len(np.where(np.abs(perm_corr) >= np.abs(true_corr))[0]) / n_iter
        
        # lower bound for perm_p: 1/n_iter
        if perm_p == 0:
            perm_p = 1/n_iter

        return perm_p

    def permutation_test_jsd(self, left, right):
        """
        To perform a permutation test for the information-theoretic measures Jensen-Shannon divergence
        and distance, the binned probabilites of left and right are shuffled, instead of the values 
        themself.
        This functions calculates the binned probabilities for the permutation test of
        Jensen-Shannon divergence/distance.
        """
        # calculate bins, in case of jensen_shannon distance left and right get the same bins
        bins = get_2D_bins(left, right, bins=self.function_args['bins'], same_bins=True)
        # calculate unconditioned histograms
        hist_left = np.histogram(left, bins=bins[0])[0]
        hist_right = np.histogram(right, bins=bins[1])[0]
        # calculate probabilities
        p_left = (hist_left / np.sum(hist_left))
        p_right = (hist_right / np.sum(hist_right))
        
        return self.permutation_test(p_left, p_right, xy_probabilities=True)


class CorrelationWarning(Base):
    """
    Warning occured during the calculation of a correlation metric.
    This warning has to be connected to one or many CorrelationMatrix 
    records.

    """
    __tablename__ = 'correlation_warnings'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    category = sa.Column(sa.String(30), nullable=False)
    message = sa.Column(sa.String, nullable=False)
    backtrace = sa.Column(sa.String, nullable=True)

    # relations
    matrix_values = relationship('CorrelationMatrix', secondary='correlation_nm_warning', back_populates='warnings')


class CorrelationWarningAssociation(Base):
    __tablename__ = 'correlation_nm_warning'

    matrix_id = sa.Column(sa.BigInteger, sa.ForeignKey('correlation_matrix.id'), primary_key=True)
    warning_id = sa.Column(sa.Integer, sa.ForeignKey('correlation_warnings.id'), primary_key=True)


class CorrelationMatrix(Base):
    """
    Correlation matrix
    """
    __tablename__ = 'correlation_matrix'

    # columns
    id = sa.Column(sa.BigInteger, primary_key=True)
    metric_id = sa.Column(sa.Integer, sa.ForeignKey('correlation_metrics.id'), nullable=False)
    value = sa.Column(sa.Numeric, nullable=False)
    p_value = sa.Column(sa.Numeric, nullable=True)
    identifier = sa.Column(sa.String(200), nullable=True)
    left_id = sa.Column(sa.Integer, nullable=False)
    right_id = sa.Column(sa.Integer, nullable=False)

    # this timestamp can be used to invalidate correlations after some time
    calculated = sa.Column(sa.DateTime, default=dt.utcnow, onupdate=dt.utcnow)

    # relationships
    metric = relationship("CorrelationMetric")
    warnings = relationship("CorrelationWarning", secondary='correlation_nm_warning', back_populates='matrix_values')

    @classmethod
    def create(
            cls,
            session: sa.orm.Session,
            entry: Union[int, str, 'Entry', 'ImmutableResultSet'],
            other: Union[int, str, 'Entry', 'ImmutableResultSet'],
            metric: Union[int, str, CorrelationMetric],
            left,
            right,
            threshold=None,
            commit=False,
            start=None,
            end=None,
            identifier=None,
            if_exists='omit',
            harmonize=True,
            p_value=True,
            force_overlap=False,
            **kwargs
        ):
        """
        Create a new matrix value for storage.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            session to the database
        entry : metacatalog.models.Entry
            Metadata entry to calculate
            Can also be of type ImmutableResultSet.
        other : metacatalog.models.Entry
            Other Metadata entry to correlate.
            Can also be of type ImmutableResultSet.
        metric : CorrelationMetric
            The id (int), symbol (str) or CorrelationMetric itself
            to be used.
        left : pandas.Series
            Left data of entry.
        right : pandas.Series
            Right data of other.
        threshold : float
            If set, any correlate absolute value lower than threshold
            will not be stored to the database. Has no effect if
            commit is False
        commit : bool
            If True, the matrix value will be persisted on creation.
            Defaults to False
        start : datetime.datetime
            Start date to filter data. If None (default), no filter
            will be applied.
        end : datetime.datetime
            End date to filter data. If None (default), no filter
            will be applied.
        if_exists : str
            Can be 'omit' or 'replace' to either skip the creation
            of a new cell or force re-calculation in case it already
            exists.
        harmonize : bool
            If True, only datapoints from left and right with matching
            indices are used for the calculation of metrics. 
            This way, the length of left and right also match.
            Defaults to True.
        p_value : bool
            If True, the p-value for the metric is saved to the database.
            The p-values are calculated from permutation tests.
            Significantly increases the calculation time.
        force_overlap : bool
            If True, the correlation metric will only be calculated
            for data of overlapping indices. If there are None,
            None is returned.
            Defaults to False.
        
        Returns
        -------
        matrix_value : CorrelationMatrix
            An object representing one cell in the CorrelationMatrix

        """
        # We need to import them here, otherwise there is a circular import if
        # metacatalog tries to load this extension, that in turn tries to load metacatalog 
        from metacatalog.models import Entry
        from metacatalog.util.results import ImmutableResultSet
        from metacatalog import api

        # check if entry is an int (id), str (uuid) or Entry
        if isinstance(entry, int):
            entry = api.find_entry(session, id=entry)[0]
        elif isinstance(entry, str):
            entry = api.find_entry(session, uuid=entry)[0]
        if not isinstance(entry, (Entry, ImmutableResultSet)):
            raise AttributeError('entry is not a valid metacatalog.Entry or metacatalog.util.results.ImmutableResultSet.')
        
        # check if other is an int (id), str (uuid) or Entry
        if isinstance(other, int):
            other = api.find_entry(session, id=other)[0]
        elif isinstance(entry, str):
            other = api.find_entry(session, uuid=other)[0]
        if not isinstance(other, (Entry, ImmutableResultSet)):
            raise AttributeError('other is not a valid metacatalog.Entry or metacatalog.util.results.ImmutableResultSet.')

        # get the metric
        if isinstance(metric, int):
            metric = session.query(CorrelationMetric).filter(CorrelationMetric.id == metric).one()
        elif isinstance(metric, str):
            metric = session.query(CorrelationMetric).filter(CorrelationMetric.symbol == metric).one()
        if not isinstance(metric, CorrelationMetric):
            raise AttributeError('metric is not a valid CorrelationMetric')

        # load existing matrix if any
        # always use the id of the first member (Entry) of an ImmutableResultSet as left_id / right_id
        if type(entry) == ImmutableResultSet:
            query = session.query(CorrelationMatrix).filter(CorrelationMatrix.left_id==entry._members[0].id)
        else:
            query = session.query(CorrelationMatrix).filter(CorrelationMatrix.left_id==entry.id)

        if type(other) == ImmutableResultSet:
            query = query.filter(CorrelationMatrix.right_id==other._members[0].id)
        else:
            query = query.filter(CorrelationMatrix.right_id==other.id)
        query = query.filter(CorrelationMatrix.metric_id==metric.id)
        
        if identifier is not None:
            query = query.filter(CorrelationMatrix.identifier == identifier)
        
        matrix = query.first()
        
        # handle omit
        if if_exists == 'omit':
            if matrix is not None and matrix.value is not None:
                return matrix

        # create an instance if needed
        if matrix is None:
            matrix = CorrelationMatrix() 

        # only proceed with the calculation if data is still available during data preprocessing:
        if (len(left) != 0 and len(right) != 0):
            # handle overlap
            # TODO - maybe we can use the TemporalExtent here to not download 
            # non-overlapping data.
            if force_overlap:
                max_start = max(right.index.min(), left.index.min())
                min_end = min(left.index.max(), right.index.max())
                left = left.loc[max_start:min_end, ].copy()
                right = right.loc[max_start:min_end, ].copy()  
                left = left.to_numpy()  
                right = right.to_numpy()  

        if (len(left) != 0 and len(right) != 0):
            # harmonize left and right data by matching indices
            if harmonize:
                harmonized_index = right[right.index.isin(left.index)].index
                left = left.loc[harmonized_index]
                right = right.loc[harmonized_index]
                if len(harmonized_index) == 0:
                    matrix.add_warning(category='HarmonizationWarning', 
                                       message='Indices of left and right data have no matches, harmonization not possible.',
                                       session=session, commit=False)

        if (len(left) != 0 and len(right) != 0):
            # pandas.Series to np.ndarray
            left = left.to_numpy()
            right = right.to_numpy()

            # if left or right is from a data source with more than one column, np.hstack converts a list of lists to a list / array 
            left = np.hstack(left)
            right = np.hstack(right)

        if (len(left) != 0 and len(right) != 0):
            # remove NaN values from both left and right (if any are included)
            if np.isnan(left).any() or np.isnan(right).any():
                nan_indices = np.logical_not(np.logical_or(np.isnan(left), np.isnan(right)))
                left = left[nan_indices]
                right = right[nan_indices]

        if (len(left) == 0 or len(right) == 0):
            matrix.add_warning(category='NoDataWarning', 
                               message='No data available for left and/or right entry due to datasource, harmonization or nan remove.',
                               session=session, commit=False)
            corr = np.nan
        else:
            # calculate, if warnings / errors occur, store them in table correlation warnings
            with warnings.catch_warnings(record=True) as w:
                try:
                    corr = metric.calc(left, right)
                except Exception as e:
                    matrix.add_warning(category=e.__class__.__name__, message=str(e), session=session, commit=False)
                    corr = np.nan

            if w:
                # use a list of unique warnings (set) (when messages occur twice in one calculation, they are also added twice in table correlation_warnings -> integrity error)
                warn_list = []
                for warn in w: warn_list.append((str(warn.category.__name__), str(warn.message)))

                for warn in set(warn_list):
                    matrix.add_warning(category=warn[0], message=warn[1], session=session, commit=False)

            # if p_value = True: calculate p-value with permutation test
            if p_value:
                try:
                    # jensen shannon distance and divergence: calculate binned probabilities (discretization), permute right probabilities
                    if 'js_d' in metric.symbol:
                        matrix.p_value = metric.permutation_test_jsd(left, right, n_iter=1000, seed=42) # set random seed: reproducibility (master thesis)
                    else:
                        matrix.p_value = metric.permutation_test(left, right, n_iter=1000, seed=42) # set random seed: reproducibility (master thesis)
                except Exception as e:
                    matrix.add_warning(category=f"Permutation warning, {e.__class__.__name__}", message=str(e), session=session, commit=False)
                    matrix.p_value = np.nan
            else:
                matrix.p_value=np.nan

        # build the matrix value
        matrix.metric_id=metric.id

        # ImmutableResultSet: use id of first member as left_id
        if type(entry) == ImmutableResultSet:
            matrix.left_id=entry._members[0].id
        else:
            matrix.left_id=entry.id
        # ImmutableResultSet: use id of first member as right_id
        if type(other) == ImmutableResultSet:
            matrix.right_id=other._members[0].id
        else:
            matrix.right_id=other.id

        matrix.value=corr
        matrix.identifier=identifier

        if commit:
            # if smaller than threshold, return anyway
            if threshold is not None and abs(corr) < threshold:
                return matrix
            
            # else add
            try:
                session.add(matrix)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e
        
        # return
        return matrix

    def add_warning(self, session, category, message, commit=False):
        """
        Add a new warning to this instance
        """
        # find or create the warning instance
        with session.no_autoflush:
            warn = session.query(CorrelationWarning).filter(CorrelationWarning.category==category, CorrelationWarning.message==message).first()
        if warn is None:
            warn = CorrelationWarning(category=category, message=message)
        
        # append warning
        self.warnings.append(warn)

        # commit if requested
        if commit:
            try:
                session.add(self)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e


def merge_declarative_base(other: sa.MetaData):
    """
    Merge this declarative base with metacatalog declarative base
    to enable foreign keys and relationships between both classes.
    """
    # build missing columns
    _connect_to_metacatalog()

    # add these tables to the other metadata
    CorrelationMetric.__table__.to_metadata(other)
    CorrelationMatrix.__table__.to_metadata(other)
    CorrelationWarning.__table__.to_metadata(other)
    CorrelationWarningAssociation.__table__.to_metadata(other)

    # TODO: here a relationship to Entry can be build if needed


def _connect_to_metacatalog():
    """
    Call this method, after the two declarative bases are connected.
    Creates missing columns and foreign keys on the tables and
    add the relationships

    """
    # add the two foreign keys to Entry
    # we need to check if the columns are already there, as the extension might already
    # be loaded by metacatalog and the connection is already there
    if not hasattr(CorrelationMatrix, 'left_id'):
        CorrelationMatrix.left_id  = sa.Column(sa.Integer, sa.ForeignKey('entries.id'), nullable=False)
    if not hasattr(CorrelationMatrix, 'right_id'):
        CorrelationMatrix.right_id = sa.Column(sa.Integer, sa.ForeignKey('entries.id'), nullable=False)
