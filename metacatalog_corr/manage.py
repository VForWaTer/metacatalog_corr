import warnings
from sqlalchemy.exc import SAWarning
from metacatalog import api

from metacatalog_corr import models
from metacatalog_corr.settings import DEFAULT_METRICS


def _connect(connection='default'):
    """
    Connect to metacatalog by merging the declarative base
    and creating Foreign Key declarations.
    Returns the merged base

    """
    # only import when needed
    from metacatalog.db.base import Base

    # merge the declarative bases
    # error the warning and ignore then
    try:
        with warnings.catch_warnings():
            # This will raise the Warning instead of printing it
            warnings.simplefilter('error', category=SAWarning)
            models.merge_declarative_base(Base.metadata)
    except SAWarning:
        # do nothing here as
        pass

    return Base


def install(connection='default', verbose=False):
    # get the merged base
    Base  = _connect(connection=connection)

    # get a database connection
    session = api.connect_database(connection)

    # install
    Base.metadata.create_all(bind=session.bind)

    # migrate default values
    migrate(connection=connection)


def migrate(connection='default', verbose=False):
    # get the merged base
    Base = _connect(connection=connection)

    # connect
    session = api.connect_database(connection)

    for default in DEFAULT_METRICS:
        symbol = default['symbol']
        metric = session.query(models.CorrelationMetric).filter(models.CorrelationMetric.symbol==symbol).first()
        if metric is not None:
            for key, val in default.items():
                setattr(metric, key, val)
        else:
            metric = models.CorrelationMetric(**default)
        
        try:
            session.add(metric)
            session.commit()
            if verbose:
                print(f'Updated {symbol}')
        except Exception as e:
            session.rollback()
            print(f'[ERROR] on {symbol}.\n{str(e)}')
    
    # done
    if verbose:
        print('Done.')
    

class ManageCli:
    def install(self, connection='default'):
        install(connection=connection, verbose=True)

    def migrate(self, connection='default'):
        migrate(connection=connection, verbose=True)


if __name__ == '__main__':
    import fire
    fire.Fire(ManageCli)