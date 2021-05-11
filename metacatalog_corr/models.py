import importlib
from datetime import datetime as dt
from typing import Union

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
import numpy as np


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
    symbol = sa.Column(sa.String(12), nullable=False)
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
        Load the actual function and return
        """
        mod = importlib.import_module(self.import_path)
        func = getattr(mod, self.function_name)
    
    def calc(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculate the metric for the given data
        """
        return self.func(left, right, **self.function_args)


class CorrelationMatrix(Base):
    """
    Correlation matrix
    """
    __tablename__ = 'correlation_matrix'

    # columns
    id = sa.Column(sa.BigInteger, primary_key=True)
    metric_id = sa.Column(sa.Integer, sa.ForeignKey('correlation_metrics.id'), nullable=False)
    value = sa.Column(sa.Numeric, nullable=False)
    identifier = sa.Column(sa.String(30), nullable=True)

    # this timestamp can be used to invalidate correlations after some time
    calculated = sa.Column(sa.DateTime, default=dt.utcnow, onupdate=dt.utcnow)

    # relationships
    metric = relationship("CorrelationMetric")

    @classmethod
    def create(
            cls,
            session: sa.orm.Session,
            entry: Union[int, str, 'Entry'],
            other: Union[int, str, 'Entry'],
            metric: Union[int, str, CorrelationMetric],
            threshold=None,
            commit=False,
            start=None,
            end=None,
            identifier=None,
            if_exists='omit',
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
        other : metacatalog.models.Entry
            Other Metadata entry to correlate
        metric : CorrelationMetric
            The id (int), symbol (str) or CorrelationMetric itself
            to be used.
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
        force_overlap : bool
            If True, the correlation metric will only be calculated
            for data of overlapping indices. If there are None,
            None is returned.
            Defaults to False
        
        Keyword Arguments
        -----------------
        left_data : numpy.ndarray
            If given, the create function will not download the data
            from metacatalog again
        right_data : numpy.ndarray
            If given, the create function will not download the data
            from metacatalog again
        
        Returns
        -------
        matrix_value : CorrelationMatrix
            An object representing one cell in the CorrelationMatrix

        """
        # We need to import them here, otherwise there is a circular import if
        # metacatalog tries to load this extension, that in turn tries to load metacatalog 
        from metacatalog.models import Entry
        from metacatalog import api

        # check if entry is an int (id), str (uuid) or Entry
        if isinstance(entry, int):
            entry = api.find_entry(session, id=entry)[0]
        elif isinstance(entry, str):
            entry = api.find_entry(session, uuid=entry)[0]
        if not isinstance(entry, Entry):
            raise AttributeError('entry is not a valid metacatalog.Entry.')
        
        # check if other is an int (id), str (uuid) or Entry
        if isinstance(other, int):
            other = api.find_entry(session, id=other)[0]
        elif isinstance(entry, str):
            other = api.find_entry(session, uuid=other)[0]
        if not isinstance(other, Entry):
            raise AttributeError('other is not a valid metacatalog.Entry.')

        # get the metric
        if isinstance(metric, int):
            metric = session.query(CorrelationMetric).filter(CorrelationMetric.id == metric).one()
        elif isinstance(metric, str):
            metric = session.query(CorrelationMetric).filter(CorrelationMetric.symbol == metric).one()
        if not isinstance(metric, CorrelationMetric):
            raise AttributeError('metric is not a valid CorrelationMetric')

        # load existing matrix if any
        query = session.query(CorrelationMatrix).filter(CorrelationMatrix.left_id==entry.id)
        query = query.filter(CorrelationMatrix.right_id==other.id)
        query = query.filter(CorrelationMatrix.metric_id==metric.id)
        
        if identifier is not None:
            query = query.filter(CorrelationMatrix.identifer == identifier)
        matrix = query.first()
        
        # handle omit
        if if_exists == 'omit':
            if matrix is not None and matrix.value is not None:
                return matrix

        # create a instance if needed
        if matrix is None:
            matrix = CorrelationMatrix() 
        
        # get the left data
        if 'left_data' in kwargs:
            left = kwargs['left_data']
        else:
            left_df = entry.get_data(start=start, end=end)
            left = left_df[entry.datasource.data_names].values

        # get the right data
        if 'right_data' in kwargs:
            right = kwargs['right_data']
        else:
            right_df = other.get_data(start=start, end=end)
            right = right_df[entry.datasource.data_names].values

        # handle overlap
        # TODO - maybe we can use the TemporalExtent here to not download 
        # non-overlapping data. 
        if force_overlap:
            max_start = max(right.index.min(), left.index.min())
            min_end = min(left.index.max(), right.index.max())
            left = left.loc[max_start:min_end, ].copy()
            right = right.loc[max_start:min_end, ].copy()

        # check if data is actually available:
        if len(left) == 0 or len(right) == 0:
            return None
       
        # calculate
        corr = metric.calc(left, right)

        # build the matrix value
        matrix.metric_id=metric.id,
        matrix.left_id=entry.id,
        matrix.right_id=other.id,
        matrix.value=corr
        matrix.identifier = identifier

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
