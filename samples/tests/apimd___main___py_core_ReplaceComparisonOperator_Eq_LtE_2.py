from importlib.machinery import SourceFileLoader


def test():
    loader = SourceFileLoader("AAAA", "apimd/__main__.py")
    module = loader.load_module()
