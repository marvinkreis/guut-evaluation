from mergesort import mergesort

def test__mergesort():
    """The mutant's change causes mergesort to fail with a single-element input."""
    output = mergesort([42])
    assert output == [42], "mergesort must correctly sort a single-element list"