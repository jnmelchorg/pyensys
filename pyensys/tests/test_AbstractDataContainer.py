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

def test_extend_op1():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2,4):
        AO2.append(str(x), x)
    AO1.extend(AO2)
    assert AO1._container == [["0", 0], ["1", 1], ["2", 2], ["3", 3]]
    
def test_extend_op2():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(2,4):
        AO2.append(str(x), x)
    AO1.extend(AO2)
    assert AO1._container == {"0": 0, "1": 1, "2": 2, "3": 3}

def test_extend_op3():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2,4):
        AO2.append(str(x), x)
    with raises(TypeError):
        AO1.extend(AO2)
    
def test_extend_op4():
    AO1 = []
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2,6):
        AO2.append(str(x), x)
    with raises(TypeError):
        AO2.extend(AO1)

def test_pop_op1():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,3):
        AO1.append(str(x), x)
    AO1.pop("1")
    assert AO1._container == {"0": 0, "2": 2}

def test_pop_op2():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,3):
        AO1.append(str(x), x)
    AO1.pop("1")
    assert AO1._container == [["0", 0], ["2", 2]]

def test_pop_op3():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,3):
        AO1.append(str(x), x)
    with raises(KeyError):
        AO1.pop("3")

def test_pop_op4():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,3):
        AO1.append(str(x), x)
    with raises(KeyError):
        AO1.pop("3")

def test_eq_op1():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(2,4):
        AO2.append(str(x), x)
    assert not AO1 == AO2

def test_eq_op2():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(0,2):
        AO2.append(str(x), x)
    assert AO1 == AO2

def test_eq_op3():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(0,2):
        AO2.append(str(x), x)
    assert AO1 == AO2

def test_eq_op4():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(0,3):
        AO2.append(str(x), x)
    assert not AO1 == AO2

def test_eq_op5():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(0,3):
        AO2.append(str(x), x)
    with raises(TypeError):
        assert AO1 == AO2

def test_eq_op6():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = []
    with raises(TypeError):
        assert AO1 == AO2

def test_ne_op1():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(2,4):
        AO2.append(str(x), x)
    assert AO1 != AO2

def test_ne_op2():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(0,2):
        AO2.append(str(x), x)
    assert not AO1 != AO2

def test_ne_op3():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(0,2):
        AO2.append(str(x), x)
    assert not AO1 != AO2

def test_ne_op4():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(0,3):
        AO2.append(str(x), x)
    assert AO1 != AO2

def test_ne_op5():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(0,3):
        AO2.append(str(x), x)
    with raises(TypeError):
        assert AO1 != AO2

def test_ne_op6():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,2):
        AO1.append(str(x), x)
    AO2 = []
    with raises(TypeError):
        assert AO1 != AO2

def test_contains_op1():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(2,4):
        AO2.append(str(x), x)
    assert AO2 in AO1

def test_contains_op2():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2,4):
        AO2.append(str(x), x)
    assert AO2 in AO1

def test_contains_op3():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(2,5):
        AO2.append(str(x), x)
    assert not AO2 in AO1

def test_contains_op4():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2,5):
        AO2.append(str(x), x)
    assert not AO2 in AO1

def test_contains_op5():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(2,4):
        AO2.append(str(x), x)
    with raises(TypeError):
        assert AO2 in AO1

def test_contains_op6():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0,4):
        AO1.append(str(x), x)
    AO2 = []
    with raises(TypeError):
        assert AO2 in AO1

def test_setitem_in_dictionary():
    AO = AbstractDataContainer()
    AO.create_dictionary()
    AO.append("0", 2)
    AO["0"] = 4
    assert AO["0"] == 4

def test_setitem_in_list():
    AO = AbstractDataContainer()
    AO.create_list()
    AO.append("0", 2)
    AO["0"] = 4
    assert AO["0"] == 4