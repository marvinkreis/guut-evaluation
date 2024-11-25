import sys


def test():
    sys.hexversion = 0x03080000
    import flutils.txtutils as txtutils
    assert txtutils.cached_property.__module__ == "functools"
