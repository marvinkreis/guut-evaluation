from flatten import flatten

def test__flatten():
    """The mutant changes the behavior of flatten such that it yields generators instead of non-list objects."""
    output = list(flatten([[1, 2], 3, 4]))
    assert all(not isinstance(item, (list, type(flatten([])))) for item in output), "flatten must yield non-list objects"