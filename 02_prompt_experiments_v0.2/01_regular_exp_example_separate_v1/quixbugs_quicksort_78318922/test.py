from quicksort import quicksort

def test__quicksort():
    """The mutant, which changes comparison from >= to >, would omit duplicates of pivot."""
    input_array = [3, 1, 2, 3]
    output = quicksort(input_array)
    assert output == [1, 2, 3, 3], f"Expected the output to include all instances of duplicates, got {output}"