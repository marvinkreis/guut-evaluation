from find_in_sorted import find_in_sorted

def test__find_in_sorted_final_not_found():
    """
    Test to verify that the find_in_sorted function works correctly for a value not present in the array. 
    Specifically checking the case with input (arr = [1, 2, 3, 4, 5], x = 6), where baseline should return -1
    and the mutant is expected to cause a RecursionError due to its modified search logic.
    """
    output = find_in_sorted([1, 2, 3, 4, 5], 6)
    assert output == -1