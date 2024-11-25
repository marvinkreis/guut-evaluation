from flake8.plugins.pyflakes import FlakesChecker


def test():
    class MockParser:
        def add_option(self, *args, **kwargs):
            if "normalize_paths" in kwargs:
                assert kwargs["normalize_paths"]

    parser = MockParser()
    FlakesChecker.add_options(parser)

