from quicksort import quicksort

def test__quicksort():
    """The change from 'greater' to '==' in quicksort should omit duplicates."""
    input_list = [5, 3, 3, 5, 2, 1, 4, 4]
    output = quicksort(input_list)
    assert output == [1, 2, 3, 3, 4, 4, 5, 5], "quicksort must handle duplicates correctly"