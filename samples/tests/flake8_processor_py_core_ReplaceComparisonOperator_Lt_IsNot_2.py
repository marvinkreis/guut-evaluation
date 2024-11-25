import warnings


def test():
    warnings.filterwarnings("error")

    # importing without __pycache__ present raises a SyntaxWarning
    from flake8.processor import FileProcessor
