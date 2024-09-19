from bucketsort import bucketsort

def test__bucketsort():
    """The current implementation of the mutant causes the output length to be incorrect."""
    output = bucketsort([3, 1, 4, 1, 5, 9, 2, 6, 5], 10)
    expected_output_count = len([3, 1, 4, 1, 5, 9, 2, 6, 5])  # This is 9
    assert len(output) == expected_output_count, "The output count must match the input count"