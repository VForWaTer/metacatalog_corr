from sqlalchemy.orm import object_session
from metacatalog import __version__
from metacatalog.ext import MetacatalogExtensionInterface
from metacatalog.models import Entry

from metacatalog_corr import models


def find_correlated_data(self, limit: int = None, return_iterator = False):
    """
    """
    raise NotImplementedError


def index_correlation_matrix(self, other: list, metrics = ['pearson']):
    """
    """
    raise NotImplementedError


class CorrExtension(MetacatalogExtensionInterface):
    """
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
