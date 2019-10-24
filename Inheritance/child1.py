from Inheritance.parent import Parent

class Child1(Parent):
    def __init__(self, name):
        super().__init__(name)

    def get_info(self):
        return self.name

    def exec_func(self):
        return;