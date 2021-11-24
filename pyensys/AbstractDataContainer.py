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
            return self._container[self._key_to_position[key]]
    
    def __iter__(self):
        if self._is_dictionary:
            self._container_iterator = iter(self._container.items())
        elif self._is_list:
            self._key_to_position_iterator = iter(self._key_to_position.items())
        return self
    
    def __next__(self):
        if self._is_dictionary:
            return next(self._container_iterator)
        elif self._is_list:
            key, value = next(self._key_to_position_iterator)
            return key, self._container[value]
    
    def __len__(self):
        return len(self._container)
    
    def create_dictionary(self):
        self._container = {}
        self._is_dictionary = True
        
    def create_list(self):
        self._container = []
        self._key_to_position = {}
        self._is_list = True

class AbstractDataContainerAppend(AbstractDataContainerBase):
    def append(self, key: str,  value: Any):
        if self._is_dictionary:
            self._container[key] = value
        elif self._is_list:
            self._key_to_position[key] = len(self._container)
            self._container.append(value)

class AbstractDataContainerGet(AbstractDataContainerAppend):
    def get(self, key: str):
        if self._is_dictionary:
            return self._container.get(key, None)
        elif self._is_list:
            if self._key_to_position.get(key, None) is not None:
                return self._container[self._key_to_position[key]]
            else:
                return None

class AbstractDataContainer(AbstractDataContainerGet):
    def __init__(self):
        super().__init__()