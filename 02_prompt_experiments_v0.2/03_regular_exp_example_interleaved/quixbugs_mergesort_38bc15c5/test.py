from mergesort import mergesort

def test__mergesort():
    """The mutant fails to handle single-element lists due to improper base case handling."""
    
    # Single element case, expected to return the same single element in a list
    output_single = mergesort([1])
    assert output_single == [1], "mergesort must return the same single element list"