
class BaseSelector:
    def __init__(self, *args, **kwargs):
        pass

    def select(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

