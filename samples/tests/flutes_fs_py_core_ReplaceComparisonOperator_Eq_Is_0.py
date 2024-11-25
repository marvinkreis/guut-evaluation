import warnings


def test():
    warnings.filterwarnings("error")
    # importing without __pycache__ present raises a SyntaxWarning
    try:
        from flutes.fs import get_folder_size
    except SyntaxError:
        assert False


