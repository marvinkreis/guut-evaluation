from mergesort import mergesort

def test__mergesort():
    """The mutant changes the base case for mergesort, which leads it to fail on non-empty input."""
    result = mergesort([5, 3])
    assert result == [3, 5], "mergesort did not sort the input correctly"