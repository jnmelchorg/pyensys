from pyensys.DataContainersInterface.OperationsAbstractDataContainer import *
from pyensys.DataContainersInterface.OperationsAbstractDataContainer import \
    _check_that_objects_are_members_of_class_abstract_data_container, _calculate_difference, \
    _calculate_difference_if_containers_are_lists, _calculate_difference_if_containers_are_dictionaries

from pytest import raises
from unittest.mock import patch


def test_calculate_difference_if_containers_are_dictionaries():
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    for x in range(0, 4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    for x in range(2, 6):
        AO2.append(str(x), x)
    AO3 = _calculate_difference_if_containers_are_dictionaries(AO1, AO2)
    assert AO3._container == {"0": 0, "1": 1}


def test_calculate_difference_if_containers_are_lists():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    for x in range(0, 4):
        AO1.append(str(x), x)
    AO2 = AbstractDataContainer()
    AO2.create_list()
    for x in range(2, 6):
        AO2.append(str(x), x)
    AO3 = _calculate_difference_if_containers_are_lists(AO1, AO2)
    assert AO3._container == [["0", 0], ["1", 1]]


@patch(
    "pyensys.DataContainersInterface.OperationsAbstractDataContainer."
    "_calculate_difference_if_containers_are_dictionaries")
def test_calculate_difference_with_dictionary_containers(mock_method):
    AO1 = AbstractDataContainer()
    AO1.create_dictionary()
    AO2 = AbstractDataContainer()
    AO2.create_dictionary()
    _calculate_difference(AO1, AO2)
    mock_method.assert_called_once()


@patch("pyensys.DataContainersInterface.OperationsAbstractDataContainer._calculate_difference_if_containers_are_lists")
def test_calculate_difference_with_list_containers(mock_method):
    AO1 = AbstractDataContainer()
    AO1.create_list()
    AO2 = AbstractDataContainer()
    AO2.create_list()
    _calculate_difference(AO1, AO2)
    mock_method.assert_called_once()


def test_type_error_in_calculate_difference():
    AO1 = AbstractDataContainer()
    AO1.create_list()
    AO2 = AbstractDataContainer()
    with raises(TypeError):
        _calculate_difference(AO1, AO2)


def test_check_that_objects_are_members_of_class_abstract_data_container():
    AO1 = AbstractDataContainer()
    AO2 = AbstractDataContainer()
    assert _check_that_objects_are_members_of_class_abstract_data_container(AO1, AO2)


def test_check_that_objects_are_not_members_of_class_abstract_data_container():
    AO1 = AbstractDataContainer()
    AO2 = []
    assert not _check_that_objects_are_members_of_class_abstract_data_container(AO1, AO2)


@patch("pyensys.DataContainersInterface.OperationsAbstractDataContainer._calculate_difference")
def test_difference_abstract_data_containers_with_correct_inputs(mock_method):
    AO1 = AbstractDataContainer()
    AO2 = AbstractDataContainer()
    difference_abstract_data_containers(AO1, AO2)
    mock_method.assert_called_once()


def test_type_error_in_difference_abstract_data_containers():
    AO1 = AbstractDataContainer()
    AO2 = []
    with raises(TypeError):
        difference_abstract_data_containers(AO1, AO2)
