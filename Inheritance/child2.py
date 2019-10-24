from Inheritance.parent import Parent

class Child2(Parent):
    def __init__(self, name):
        super().__init__(name)

    def get_info(self):
        return self.name

    def exec_func(self):
        return;
    
    def get_salutation(self):
        return 'Hello ' + self.name
