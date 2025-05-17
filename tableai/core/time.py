from dataclasses import dataclass, field
from datetime import datetime, timezone
import abc
from typing import List, Dict, Any
from uuid import UUID, uuid4


# -----------------------------------------
# Abstract Time Utility
# -----------------------------------------
class AbstractTimeUtil(abc.ABC):
    @abc.abstractmethod
    def now_utc(self) -> datetime:
        pass

    @abc.abstractmethod
    def to_iso_format(self, dt: datetime) -> str:
        pass

    @abc.abstractmethod
    def from_iso_format(self, time_str: str) -> datetime:
        pass


# -----------------------------------------
# Concrete Implementation of the Time Utility
# -----------------------------------------
class UtcTimeUtil(AbstractTimeUtil):
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def to_iso_format(self, dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def from_iso_format(self, time_str: str) -> datetime:
        return datetime.fromisoformat(time_str)


# -----------------------------------------
# A Generic TimeTrackedMixin
# -----------------------------------------
@dataclass
class TimeTrackedMixin:
    """
    A generic mixin that can handle any number of datetime fields.
    Subclasses must define:
       _time_fields = ['field_name1', 'field_name2', ...]
    and corresponding dataclass fields.
    """
    time_util: AbstractTimeUtil = field(default_factory=UtcTimeUtil)

    # Each subclass should define a class-level list of fields to manage:
    # _time_fields = ['created_at', 'updated_at'] etc.

    def __post_init__(self):
        # Ensure all time fields are initialized and timezone-aware.
        # If a field is None, initialize it to now_utc().
        for field_name in getattr(self, '_time_fields', []):
            dt = getattr(self, field_name, None)
            if dt is None:
                dt = self.time_util.now_utc()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            setattr(self, field_name, dt)

    def to_dict(self) -> dict:
        # Convert all defined time fields to ISO strings
        data = {}
        for field_name in getattr(self, '_time_fields', []):
            dt = getattr(self, field_name)
            data[field_name] = self.time_util.to_iso_format(dt)
        return data

    @classmethod
    def from_dict(cls, data: dict):
        # Parse all defined time fields from dictionary
        time_util = UtcTimeUtil()
        kwargs = {}
        for field_name in getattr(cls, '_time_fields', []):
            if field_name in data:
                kwargs[field_name] = time_util.from_iso_format(data[field_name])
        kwargs['time_util'] = time_util
        return cls(**kwargs)


# -----------------------------------------
# Specific Mixin Implementations
# -----------------------------------------
@dataclass
class Created(TimeTrackedMixin):
    """
    Mixin that adds created_at and updated_at fields.
    """
    _time_fields = ['created_at', 'updated_at']
    created_at: datetime = field(default=None)
    updated_at: datetime = field(default=None)

    def update_timestamp(self):
        self.updated_at = self.time_util.now_utc()


@dataclass
class Sync(TimeTrackedMixin):
    """
    Mixin that adds a last_sync field.
    """
    _time_fields = ['last_sync']
    last_sync: datetime = field(default=None)

    def sync_now(self):
        self.last_sync = self.time_util.now_utc()


# -----------------------------------------
# Composing Mixin into a Full Model
# -----------------------------------------
@dataclass
class TimeModel(Created, Sync):
    """
    Example of combining multiple time mixes:
    This inherits from Created and Sync, so it has:
       created_at, updated_at, and last_sync
    """
    pass
