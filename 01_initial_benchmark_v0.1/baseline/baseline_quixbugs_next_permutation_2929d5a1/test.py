from next_permutation import next_permutation

def test_next_permutation():
    # Test case for a standard increasing input
    perm = [1, 2, 3]
    expected_output = [1, 3, 2]  # The next lexicographical permutation
    assert next_permutation(perm) == expected_output, "Test failed for input [1, 2, 3]"

    # Mid-range case
    perm = [3, 2, 4, 1]
    expected_output = [3, 4, 1, 2]  # The next permutation in order
    assert next_permutation(perm) == expected_output, "Test failed for input [3, 2, 4, 1]"

    # Highest permutation input
    perm = [4, 3, 2, 1]
    result = next_permutation(perm)  # Capture the return
    assert result is None, f"Test failed for input highest permutation [4, 3, 2, 1]. Expected None, but got: {result}"

    # Single element case
    perm = [1]
    result = next_permutation(perm)  # Should also return the same
    expected_output = None  # Assuming single element might just wrap around to itself properly
    assert result is None, "Test failed for input [1] (should return None)"

    print("All tests passed!")
