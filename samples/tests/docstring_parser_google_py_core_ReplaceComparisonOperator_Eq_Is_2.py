def test():
    import warnings
    warnings.filterwarnings("error")

    try:
        # importing without __pycache__ present raises a SyntaxWarning
        from docstring_parser.google import GoogleParser
    except SyntaxError:
        assert False

