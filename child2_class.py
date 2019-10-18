from parent_class import parent_class

class child2_class(parent_class):
    """description of class"""

    def __init__(self, name):
        super().__init__(name)

    def get_info(self):
        return self.name

    def exec_func(self):
        return;
    
    def get_salutation(self):
        return 'Hello ' + self.name
