from hanoi import hanoi

def test_hanoi():
    # Test case with height 2
    expected_output_2 = [(1, 2), (1, 3), (2, 3)]  # Moves to transfer 2 disks from peg 1 to peg 3
    actual_output_2 = hanoi(2, start=1, end=3)

    # Assert that the output is as expected
    assert actual_output_2 == expected_output_2, f"Expected {expected_output_2}, but got {actual_output_2}"

    # Test case with height 3
    expected_output_3 = [(1, 3), (1, 2), (3, 2), (1, 3), (2, 1), (2, 3), (1, 3)]  # Correct output for height 3
    actual_output_3 = hanoi(3, start=1, end=3)

    # Assert that the output is as expected
    assert actual_output_3 == expected_output_3, f"Expected {expected_output_3}, but got {actual_output_3}"

    # Additional checks to differentiate mutants
    assert len(actual_output_2) == 3, f"Incorrect number of moves for height 2: {len(actual_output_2)}"
    assert len(actual_output_3) == 7, f"Incorrect number of moves for height 3: {len(actual_output_3)}"