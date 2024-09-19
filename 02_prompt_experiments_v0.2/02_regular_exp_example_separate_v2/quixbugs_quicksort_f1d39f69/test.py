from quicksort import quicksort

def test__quicksort():
    """Mutant will not return all duplicate elements when the '>= pivot' is changed to '> pivot'."""
    output = quicksort([3, 3, 2, 1, 3])
    assert output == [1, 2, 3, 3, 3], "Should have returned all 3's"
    
    output = quicksort([5, 5, 10, 5])
    assert output == [5, 5, 5, 10], "Should have returned all 5's"