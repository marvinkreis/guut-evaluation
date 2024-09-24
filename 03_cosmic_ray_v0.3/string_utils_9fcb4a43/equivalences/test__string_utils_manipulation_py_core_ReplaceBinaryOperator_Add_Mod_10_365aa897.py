from string_utils import prettify

def test():
    try:
        prettify("a(b)c")
    except TypeError:
        assert False
