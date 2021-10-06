from SIRD.util import get_date_format
from dataclasses import dataclass, field
from datetime import datetime, date

import hashlib


@dataclass
class PredictInfo:
    y_frames: int = None
    _y_frames: int = field(default=False)
    test_start: datetime = None
    _test_start: datetime = field(init=False, repr=False)
    test_end: datetime = None
    _test_end: datetime = field(init=False, repr=False)

    def __init__(self, y_frames: int, test_start=None, test_end=None):
        self.y_frames = y_frames
        self.test_start = test_start
        self.test_end = test_end

    def __repr__(self):
        representation = f'DatasetInfo(y_frames: {self._y_frames}, '
        representation += f'test_start: {self._test_start}, test_end: {self._test_end})'
        return representation

    @property
    def y_frames(self) -> int:
        return self._y_frames

    @y_frames.setter
    def y_frames(self, y_frames: int):
        self._y_frames = y_frames

    @property
    def test_start(self):
        if hasattr(self, '_test_start'):
            return self._test_start
        else:
            return None

    def start_tostring(self, format: str = '%y%m%d'):
        if hasattr(self, '_test_start'):
            return self._test_start.strftime(format)
        else:
            return ''

    @test_start.setter
    def test_start(self, test_start):
        if test_start is None:
            self._test_start = datetime.now().date()

        if isinstance(test_start, str):
            format = get_date_format(test_start)
            self._test_start = datetime.strptime(test_start, format).date()
        elif isinstance(test_start, datetime):
            self._test_start = test_start.date()
        elif isinstance(test_start, date):
            self._test_start = test_start

    @property
    def test_end(self):
        if hasattr(self, '_test_end'):
            return self._test_end
        else:
            return None

    def end_tostring(self, format: str = '%y%m%d'):
        if hasattr(self, '_test_end'):
            return self._test_end.strftime(format)
        else:
            return ''

    @test_end.setter
    def test_end(self, test_end):
        if test_end is None:
            self._test_end = datetime.now().date()

        if isinstance(test_end, str):
            format = get_date_format(test_end)
            self._test_end = datetime.strptime(test_end, format).date()
        elif isinstance(test_end, datetime):
            self._test_end = test_end.date()
        elif isinstance(test_end, date):
            self._test_end = test_end

    def get_hash(self):
        hash_key = hashlib.sha1(self.__repr__().encode()).hexdigest()[:6]
        return hash_key
