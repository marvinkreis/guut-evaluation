from hanoi import hanoi

def test__hanoi():
    """The mutant changes the move from (start, end) to (start, helper), which leads to incorrect moves."""
    # Expected moves for hanoi(2)
    expected_output = [(1, 2), (1, 3), (2, 3)]
    output = hanoi(2)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"