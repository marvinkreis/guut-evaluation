from bucketsort import bucketsort

def test__bucketsort_general_case():
    """
    Test the bucketsort function with a general unsorted list.
    The input is [3, 1, 4, 2, 0]. The expected sorted output is [0, 1, 2, 3, 4].
    The mutant incorrectly processes this input due to using the wrong loop, 
    which leads to a different and incorrect output.
    """
    arr = [3, 1, 4, 2, 0]  # unsorted input
    k = 5  # upper bound
    output = bucketsort(arr, k)
    assert output == [0, 1, 2, 3, 4], f"Expected [0, 1, 2, 3, 4], got {output}"