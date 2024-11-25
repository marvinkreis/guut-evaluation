import warnings

def test():
    # importing without __pycache__ present raises a SyntaxWarning
    warnings.filterwarnings("error")
    import flake8.checker
