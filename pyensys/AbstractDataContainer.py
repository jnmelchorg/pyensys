from typing import Any

class AbstractDataContainerBase:
    def __init__(self):
        self._container = None
        self._is_dictionary = False
        self._is_list = False
        self._key_to_position = None
    
    def __getitem__(self, key: str):
        if self._is_dictionary:
            return self._container[key]
        elif self._is_list:
            for item in self._container:
                if item[0] == key:
                    return item[1]
    
    def __iter__(self):
        if self._is_dictionary:
            self._container_iterator = iter(self._container.items())
        elif self._is_list:
            self._container_iterator = iter(self._container)
        return self
    
    def __next__(self):
        if self._is_dictionary:
            return next(self._container_iterator)
        elif self._is_list:
            item = next(self._container_iterator)
            return item[0], item[1]
    
    def __len__(self):
        return len(self._container)
    
    def create_dictionary(self):
        self._container = {}
        self._is_dictionary = True
        
    def create_list(self):
        self._container = []
        self._is_list = True

class AbstractDataContainerAppend(AbstractDataContainerBase):
    def append(self, key: str,  value: Any):
        if self._is_dictionary:
            self._container[key] = value
        elif self._is_list:
            self._container.append([key, value])

class AbstractDataContainerGet(AbstractDataContainerAppend):
    def get(self, key: str):
        if self._is_dictionary:
            return self._container.get(key, None)
        elif self._is_list:
            for item in self._container:
                if item[0] == key:
                    return item[1]
            return None

class AbstractDataContainerGetAllValuesAsList(AbstractDataContainerGet):
    def get_values_as_list(self) -> list:
        if self._is_dictionary:
            return list(self._container.values())
        elif self._is_list:
            return [item[1] for item in self._container]    
class AbstractDataContainer(AbstractDataContainerGetAllValuesAsList):
    def __init__(self):
        super().__init__()


def difference_abstract_data_containers(abstract_data_container1: AbstractDataContainer, \
    abstract_data_container2:AbstractDataContainer) -> list:
    if isinstance(abstract_data_container1, AbstractDataContainer) and \
        isinstance(abstract_data_container2, AbstractDataContainer):
        if abstract_data_container1._is_dictionary and abstract_data_container2._is_dictionary:
            for key, val in abstract_data_container2._container.items():
                if abstract_data_container1._container.get(key, None) is not None and \
                    abstract_data_container1._container[key] == val:
                    abstract_data_container1._container.pop(key)
        elif abstract_data_container1._is_list and abstract_data_container2._is_list:
            abstract_data_container1._container = [item1 for item1 in \
                abstract_data_container1._container if item1 not in abstract_data_container2._container]
        else:
            raise TypeError
        return abstract_data_container1
    else:
        raise TypeError