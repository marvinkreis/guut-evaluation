from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted_mutant_killing():
    """
    Test cases to verify the correct behavior of the find_first_in_sorted function.
    This will demonstrate that the mutant with 'while lo <= hi:' causes an
    IndexError for certain inputs, whereas the baseline implementation does not.
    """
    arr = [1, 2, 3, 4, 5]

    # Case where x is not present, expecting -1
    result_not_present = find_first_in_sorted(arr, 6)
    print(f"Output for x=6 (not present): {result_not_present}")
    assert result_not_present == -1

    # Case where x is present, expecting index 2
    result_present = find_first_in_sorted(arr, 3)
    print(f"Output for x=3 (present): {result_present}")
    assert result_present == 2

    # Case for first element, expecting index 0
    result_first_element = find_first_in_sorted(arr, 1)
    print(f"Output for x=1 (first element): {result_first_element}")
    assert result_first_element == 0

    # Case for empty array, expecting -1
    result_empty = find_first_in_sorted([], 3)
    print(f"Output for empty array x=3: {result_empty}")
    assert result_empty == -1