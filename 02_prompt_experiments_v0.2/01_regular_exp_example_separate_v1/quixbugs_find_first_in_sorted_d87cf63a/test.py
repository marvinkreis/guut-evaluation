from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted():
    # Case 1: Searching for a value greater than the max
    # For the correct implementation, this should return -1
    output = find_first_in_sorted([1, 2, 3, 4, 5], 6)
    assert output == -1, "Should return -1 for value not in the list."

    # Case 2: Searching for the maximum value in the list
    # This should return the index of the value (4)
    output = find_first_in_sorted([1, 2, 3, 4, 5], 5)
    assert output == 4, "Should return index of 5."

    # Case 3: Searching in an empty list
    # This should return -1 for an empty array input
    output = find_first_in_sorted([], 1)
    assert output == -1, "Should return -1 for value not found in the empty list."

    # Case 4: Searching for an element in a list where the element is present only once
    # Should return the correct index for a single entry
    output = find_first_in_sorted([1], 1)
    assert output == 0, "Should return index of the only element."

    print("All tests passed!")

# The mutant should fail on cases that lead to out-of-bounds errors, especially on the empty list 
# and when searching for a value greater than the max.