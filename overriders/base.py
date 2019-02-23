from torch.nn import Parameter

class CustomWrapperBase(Parameter):
    def __new__(cls, data=None, requires_grad=True):
        cls._data = data
        return super(CustomWrapperBase, cls).__new__(cls, data, requires_grad=requires_grad)

    @property
    def data(self):
        return self.wrapper(self._data)

    def wrapper(self, value):
        return value

