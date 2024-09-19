from mergesort import mergesort

def test__mergesort_single_element():
    """The mutant fails to return the input list for a single-element input."""
    output = mergesort([5])
    assert output == [5], "mergesort must return the same single element when given a one-item list."