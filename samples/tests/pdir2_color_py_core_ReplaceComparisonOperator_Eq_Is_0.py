import warnings


def test():
    warnings.filterwarnings("error")
    try:
        from pdir.color import _Color
    except SyntaxError:
        assert False
