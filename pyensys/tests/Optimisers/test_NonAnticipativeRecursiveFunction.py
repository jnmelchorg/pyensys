from multiprocessing import pool
from pyensys.Optimisers.NonAnticipativeRecursiveFunction import NonAnticipativeRecursiveFunction
from pyensys.Optimisers.RecursiveFunction import InterIterationInformation, BinaryVariable
from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer

from unittest.mock import MagicMock

def test_verify_feasibility_of_sucessor_with_no_new_interventions():
    RF = NonAnticipativeRecursiveFunction()
    RF._operational_check = MagicMock()
    RF._is_opf_feasible = MagicMock(return_value=True)
    assert RF._verify_feasibility_of_sucessor_with_no_new_interventions(InterIterationInformation())
    RF._operational_check.assert_called_once()

def _create_dummy_information_to_test_get_available_interventions_for_current_year() -> \
    InterIterationInformation:
    info = InterIterationInformation()
    variables = AbstractDataContainer()
    variables.create_list()
    variables.append("0", 0)
    variables.append("1", 1)
    info.candidate_interventions.create_list()
    info.candidate_interventions.append("0", variables)
    info.candidate_interventions.append("1", AbstractDataContainer())
    info.new_interventions.create_list()
    variables = AbstractDataContainer()
    variables.create_list()
    variables.append("2", 2)
    return info

def _create_dummy_pool_of_interventions_to_test_get_available_interventions_for_current_year() -> \
    AbstractDataContainer:
    pool_interventions = AbstractDataContainer()
    pool_interventions.create_list()
    pool_interventions.append("0", 0)
    pool_interventions.append("1", 1)
    pool_interventions.append("2", 2)
    pool_interventions.append("3", BinaryVariable(installation_time=2))
    pool_interventions.append("4", BinaryVariable(installation_time=0))
    pool_interventions.append("5", BinaryVariable(installation_time=1))

    return pool_interventions

def test_get_available_interventions_for_current_year():
    info = _create_dummy_information_to_test_get_available_interventions_for_current_year()

