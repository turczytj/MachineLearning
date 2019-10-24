from abc import ABC, abstractmethod

class Parent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def exec_func(self):
        pass

    def get_salutation(self):
        return 'Greetings ' + self.name
