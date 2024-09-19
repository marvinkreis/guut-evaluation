from mergesort import mergesort

def test__mergesort():
    """Mutant fails when processing single-element lists, resulting in endless recursion."""
    
    # Test with a single element
    single_element_input = [5]
    single_element_output = mergesort(single_element_input)
    assert single_element_output == [5], "Single element list must return the same element."

    # Test with an empty array
    empty_input = []
    empty_output = mergesort(empty_input)
    assert empty_output == [], "Empty list must return an empty list."