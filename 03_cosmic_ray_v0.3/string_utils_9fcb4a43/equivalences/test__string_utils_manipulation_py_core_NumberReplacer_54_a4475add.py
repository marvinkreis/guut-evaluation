from string_utils.manipulation import __StringCompressor

def test():
    c = __StringCompressor()
    try:
        c.compress("input")
    except ValueError:
        assert False
