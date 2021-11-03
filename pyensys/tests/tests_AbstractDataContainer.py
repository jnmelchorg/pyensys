from pyensys.AbstractDataContainer import AbstractDataContainer

def test_abstract_data_container_getitem_dict():
    data = AbstractDataContainer()
    data.create_dictionary()
    data.append("1", 1)
    data.append("2", 2)
    assert data["2"] == 2

def test_abstract_data_container_iterator_dict():
    data = AbstractDataContainer()
    data.create_dictionary()
    data.append("1", 1)
    data.append("2", 2)
    result = []
    EXPECTED_RESULT = [["1", 1] , ["2", 2]]
    for key, value in data:
        result.append([key, value])
    assert result == EXPECTED_RESULT

def test_abstract_data_container_iterator_list():
    data = AbstractDataContainer()
    data.create_list()
    data.append("1", 1)
    data.append("2", 2)
    result = []
    EXPECTED_RESULT = [["1", 1] , ["2", 2]]
    for key, value in data:
        result.append([key, value])
    assert result == EXPECTED_RESULT