from __future__ import annotations
from typing import Any, Sequence, List
from pandas import DataFrame as df
from pandas import read_excel


class DataFrame:
    def __init__(self,
                 data: Any = None):
        self._data = df(data)

    def size(self) -> int:
        return self._data.size

    def get_column_headers(self) -> List[Any]:
        return self._data.columns.values.tolist()


class SpreadSheet:
    def __init__(self):
        self._data = DataFrame()

    def read_spreadsheet(self,
                         path_to_file: Any,
                         sheet_name: str | int | list[str | int] | None = 0,
                         header: int | Sequence[int] | None = None
                         ) -> DataFrame | dict[str | int, DataFrame]:
        self._data = DataFrame(read_excel(path_to_file, sheet_name, header))
        return self._data
