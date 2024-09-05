from quicksort import quicksort

def test__quicksort():
    # This test will fail on the mutant version due to improper handling of duplicates.
    assert quicksort([3, 3, 2]) == [2, 3, 3], "QuickSort must handle duplicates correctly"
    assert quicksort([1, 2, 2]) == [1, 2, 2], "QuickSort must maintain order with duplicates"
    assert quicksort([1, 1, 1, 1]) == [1, 1, 1, 1], "QuickSort must return correct order for all duplicates"