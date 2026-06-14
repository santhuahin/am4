from __future__ import annotations
import typing
from . import utils
__all__: list[str] = ['DatabaseException', 'init', 'utils']
class DatabaseException(Exception):
    pass
    ...
def init(home_dir: str | None = None) -> None:
    ...
