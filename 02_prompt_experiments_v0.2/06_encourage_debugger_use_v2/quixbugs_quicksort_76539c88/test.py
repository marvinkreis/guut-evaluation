from quicksort import quicksort

def test__quicksort():
    """The mutant excludes elements equal to the pivot, leading to missing duplicates."""
    input_data = [3, 3, 1, 2]
    output = quicksort(input_data)
    assert len(output) >= len(input_data), "quicksort must retain all elements including duplicates"