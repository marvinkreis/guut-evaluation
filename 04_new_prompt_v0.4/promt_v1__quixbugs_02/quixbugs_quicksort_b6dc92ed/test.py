from quicksort import quicksort

def test__quicksort_with_duplicates():
    """
    This test checks that the quicksort function correctly sorts an array with duplicate values.
    The input [3, 1, 2, 3] includes duplicates of the number 3. The baseline implementation should
    return [1, 2, 3, 3] while the mutant implementation should return [1, 2, 3], hence identifying
    the mutant's handling of duplicates as incorrect.
    """
    input_arr = [3, 1, 2, 3]
    output = quicksort(input_arr)
    assert output == [1, 2, 3, 3], f"Expected [1, 2, 3, 3], but got {output}"