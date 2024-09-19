from flatten import flatten

def test__flatten():
    """The mutant changes 'yield x' to 'yield flatten(x)', returning generators instead of non-list objects."""
    output = list(flatten([[1, [], [2, 3]], [[4]], 5]))
    assert all(isinstance(x, int) for x in output), "flatten must yield integer values, not generators"