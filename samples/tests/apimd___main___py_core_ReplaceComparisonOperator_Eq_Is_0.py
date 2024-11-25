def test():
    import warnings
    warnings.filterwarnings("error")

    try:
        # importing without __pycache__ present raises a SyntaxWarning
        from apimd.__main__ import main
    except SyntaxError:
        assert False
