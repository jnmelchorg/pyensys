import pytest

from pyensys.wrappers.PandasWrapper import DataFrame, SpreadSheet


def test_size():
    df = DataFrame()
    assert df.size() == 0


def test_initialise_with_dict():
    df = DataFrame(data={"Col1": [1]})
    assert df.size() == 1


def test_get_column_headers():
    df = DataFrame(data={"Col1": [1], "Col2": [2]})
    assert df.get_column_headers() == ["Col1", "Col2"]


class TestSpreadSheet:
    EXCEL_PATH = "C:\\Users\\f09903jm\\git projects\\pyensys\\pyensys\\tests\\excel\\pandas_excel_test.xlsx"
    ODS_PATH = "C:\\Users\\f09903jm\\git projects\\pyensys\\pyensys\\tests\\ods\\pandas_ods_test.ods"

    @pytest.mark.parametrize("sheet_name, header, expected_size, expected_headers, file_path",
                             [(0, None, 1, [0], EXCEL_PATH), ("Sheet2", 0, 2, ["label"], EXCEL_PATH),
                              ("Sheet3", 1, 2, ["label", "label1"], EXCEL_PATH), (0, None, 1, [0], ODS_PATH),
                             ("Sheet2", 0, 2, ["label"], ODS_PATH), ("Sheet3", 1, 2, ["label", "label1"], ODS_PATH)])
    def test_read_spreadsheet(self, sheet_name, header, expected_size, expected_headers, file_path):
        ss = SpreadSheet()
        df = ss.read_spreadsheet(path_to_file=file_path, sheet_name=sheet_name, header=header)
        assert df.size() == expected_size
        assert df.get_column_headers() == expected_headers
