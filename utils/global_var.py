def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner


@singleton
class GlobalVar(object):
    def __init__(self):
        self.__global_var = {}
        pass

    def set_var(self, key, var):
        self.__global_var[key] = var

    def get_var(self, key):
        try:
            return self.__global_var[key]
        except KeyError:
            print('GlobalVar dict had not this key')


if __name__ == '__main__':
    print('test global var')
    cls1 = GlobalVar()
    cls2 = GlobalVar()
    print(id(cls1) == id(cls2))
    cls1.set_var('first_var', 10)
    print(cls1.get_var('first_var'))
