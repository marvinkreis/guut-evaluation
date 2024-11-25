from dataclasses_json.undefined import _UndefinedParameterAction


def test():
    class A(_UndefinedParameterAction):
        pass

    try:
        A()
        assert False
    except TypeError:
        pass
