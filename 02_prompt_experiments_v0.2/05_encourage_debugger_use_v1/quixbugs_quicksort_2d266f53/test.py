from quicksort import quicksort

def test__quicksort():
    """The mutant fails to include equal elements in the sorted output, excluding duplicates."""
    input_array = [3, 1, 2, 3]
    output = quicksort(input_array)
    assert output == [1, 2, 3, 3], "quicksort must sort the list correctly with duplicates included"
    
    input_array = [5, 5, 2, 1, 5]
    output = quicksort(input_array)
    assert output == [1, 2, 5, 5, 5], "quicksort must sort the list correctly with duplicates included"