def test():
    import warnings
    warnings.filterwarnings("error")

    try:
        # importing without __pycache__ present raises a SyntaxWarning
        from apimd.compiler import get_name
    except SyntaxError:
        assert False
