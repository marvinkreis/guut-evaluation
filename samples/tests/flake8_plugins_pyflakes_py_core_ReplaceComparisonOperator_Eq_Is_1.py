import warnings


def test():
    warnings.filterwarnings("error")
    # importing without __pycache__ present raises a SyntaxWarning
    import flake8.plugins.pyflakes
