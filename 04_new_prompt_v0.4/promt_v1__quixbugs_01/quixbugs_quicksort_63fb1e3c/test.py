from quicksort import quicksort

def test__quicksort_duplicates():
    """
    Test the quicksort function with a list containing duplicate elements. 
    The input [3, 2, 1, 3, 2] includes duplicates, and the expectation is that the sorted output must include all occurrences of values. 
    The Baseline should return [1, 2, 2, 3, 3], while the Mutant will likely return [1, 2, 3], thus failing the test.
    """
    output = quicksort([3, 2, 1, 3, 2])
    assert output == [1, 2, 2, 3, 3]