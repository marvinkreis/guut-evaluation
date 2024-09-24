import sys

def test():
    # Intercept prints
    prints = []
    def intercept(fun):
        def intercepter(*args, **kwargs):
            prints.append((args, kwargs))
            fun(*args, **kwargs)
        return intercepter
    sys.stdout.write = intercept(sys.stdout.write)
    sys.stderr.write = intercept(sys.stderr.write)

    from string_utils._regex import PRETTIFY_RE
    assert not prints
