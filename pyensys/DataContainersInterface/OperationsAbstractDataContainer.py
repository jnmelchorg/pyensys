from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer

from copy import deepcopy
from typing import List


def difference_abstract_data_containers(abstract_data_container1: AbstractDataContainer,
                                        abstract_data_container2: AbstractDataContainer) -> AbstractDataContainer:
    if _check_that_objects_are_members_of_class_abstract_data_container(abstract_data_container1,
                                                                        abstract_data_container2):
        return _calculate_difference(abstract_data_container1, abstract_data_container2)
    else:
        raise TypeError


def _check_that_objects_are_members_of_class_abstract_data_container(abstract_data_container1: AbstractDataContainer,
                                                                     abstract_data_container2: AbstractDataContainer) \
        -> bool:
    return isinstance(abstract_data_container1, AbstractDataContainer) and \
           isinstance(abstract_data_container2, AbstractDataContainer)


def _calculate_difference(abstract_data_container1: AbstractDataContainer,
                          abstract_data_container2: AbstractDataContainer) -> AbstractDataContainer:
    if abstract_data_container1.is_dictionary() and abstract_data_container2.is_dictionary():
        return _calculate_difference_if_containers_are_dictionaries(abstract_data_container1, abstract_data_container2)
    elif abstract_data_container1.is_list() and abstract_data_container2.is_list():
        return _calculate_difference_if_containers_are_lists(abstract_data_container1, abstract_data_container2)
    else:
        raise TypeError


def _calculate_difference_if_containers_are_lists(abstract_data_container1: AbstractDataContainer,
                                                  abstract_data_container2: AbstractDataContainer) -> \
        AbstractDataContainer:
    difference = deepcopy(abstract_data_container1)
    difference._container = [item1 for item1 in \
                             abstract_data_container1._container if item1 not in abstract_data_container2._container]
    return difference


def _calculate_difference_if_containers_are_dictionaries(abstract_data_container1: AbstractDataContainer,
                                                         abstract_data_container2: AbstractDataContainer) -> \
        AbstractDataContainer:
    difference = deepcopy(abstract_data_container1)
    for key, val in abstract_data_container2._container.items():
        if abstract_data_container1._container.get(key, None) is not None and \
                abstract_data_container1._container[key] == val:
            difference._container.pop(key)
    return difference


def get_indexes_of_ordered_items_related_to_the_input_ordered_items(ordered_items: AbstractDataContainer,
                                                                    input_items: AbstractDataContainer) -> List[str]:
    return []
