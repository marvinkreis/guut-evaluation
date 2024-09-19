from pascal import pascal

def test_pascal():
    # Test case for n = 5
    expected_output = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    actual_output = pascal(5)
    assert actual_output == expected_output, f"Expected {expected_output}, but got {actual_output}"

    # Test case for n = 1
    expected_output_single = [[1]]
    actual_output_single = pascal(1)
    assert actual_output_single == expected_output_single, f"Expected {expected_output_single}, but got {actual_output_single}"