from quicksort import quicksort

def test__quicksort():
    """The mutant changes > to >= which omits duplicates in the sorting process."""
    test_input = [3, 6, 8, 3, 2, 7, 5, 3]
    output = quicksort(test_input)
    
    # Check length and content to ensure duplicates are preserved
    assert len(output) == 8, "quicksort must include all elements, including duplicates"
    assert output == [2, 3, 3, 3, 5, 6, 7, 8], "quicksort result must be sorted correctly with duplicates"