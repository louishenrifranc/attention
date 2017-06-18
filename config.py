from types import SimpleNamespace


class Config(SimpleNamespace):
    """
    Basic config class.  Loads a json and presents as an object structure
    """

    def __init__(self, data):
        """
        args:
            filename - the json file to load
        """
        super().__init__()
        self.__dict__.update(data.__dict__)

    def __setattr__(self, key, value):
        raise TypeError("Objects of type Config are immutable")
