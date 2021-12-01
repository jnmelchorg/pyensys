from pyensys.AbstractDataContainer import *
from pytest import raises

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

def test_abstract_data_container_get_dict_op1():
    data = AbstractDataContainer()
    data.create_dictionary()
    data.append("1", 1)
    data.append("2", 2)
    assert data.get("1") == 1

def test_abstract_data_container_get_dict_op2():
    data = AbstractDataContainer()
    data.create_dictionary()
    data.append("1", 1)
    data.append("2", 2)
    assert data.get("3") == None

def test_abstract_data_container_get_list_op1():
    data = AbstractDataContainer()
    data.create_list()
    data.append("1", 1)
    data.append("2", 2)
    assert data.get("1") == 1

def test_abstract_data_container_get_list_op2():
    data = AbstractDataContainer()
    data.create_list()
    data.append("1", 1)
    data.append("2", 2)
    assert data.get("3") == None

def test_get_values_as_list_from_dict():
    data = AbstractDataContainer()
    data.create_dictionary()
    data.append("1", 1)
    data.append("2", 2)
    assert data.get_values_as_list() == [1, 2]

def test_get_values_as_list_from_list():
    data = AbstractDataContainer()
    data.create_list()
    data.append("1", 1)
    data.append("2", 2)
    assert data.get_values_as_list() == [1, 2]

def test_get_set_difference_op1():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2,6):
        AO2.append(str(x), x)
    difference_abstract_data_containers(AO1, AO2)
    assert AO1._container == [["0", 0], ["1", 1]]

def test_get_set_difference_op2():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(2,6):
        AO2.append(str(x), x)
    difference_abstract_data_containers(AO1, AO2)
    assert AO1._container == {"0": 0, "1": 1}

def test_get_set_difference_op3():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2,6):
        AO2.append(str(x), x)
    with raises(TypeError):
        difference_abstract_data_containers(AO1, AO2)

def test_get_set_difference_op4():
    AO1 = []
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2,6):
        AO2.append(str(x), x)
    with raises(TypeError):
        difference_abstract_data_containers(AO1, AO2)
    
