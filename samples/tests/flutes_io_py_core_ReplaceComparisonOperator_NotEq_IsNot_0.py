import warnings


def test():
    warnings.filterwarnings("error")
    # importing without __pycache__ present raises a SyntaxWarning
    from flutes.io import _ReverseReadlineFile
