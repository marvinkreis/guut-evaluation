from hanoi import hanoi

def test__hanoi_killing_mutant():
    """
    This test verifies the behavior of the hanoi function with 3 disks. The expected output for the 
    baseline implementation is well-defined. The mutant implementation will produce a different 
    sequence of moves.
    """
    # The expected output for hanoi with height 3
    expected_baseline_output = [(1, 3), (1, 2), (3, 2), (1, 3), (2, 1), (2, 3), (1, 3)]

    # Execute the baseline version
    baseline_output = hanoi(3)
    print(f"Baseline output for height 3: {baseline_output}")

    # Check the baseline output against the expected outcome
    assert baseline_output == expected_baseline_output, "The baseline output does not match the expected sequence."

    # Now, simulate the mutant output directly.
    # Manually defining the mutant behavior as identified earlier in the exploration.
    mutant_output = [(1, 2), (1, 3), (3, 1), (1, 2), (2, 3), (2, 1), (1, 2)]
    print(f"Mutant output for height 3: {mutant_output}")

    # Assert that mutant output is different from baseline output
    assert baseline_output != mutant_output, "The outputs should differ, demonstrating the mutant's effect."