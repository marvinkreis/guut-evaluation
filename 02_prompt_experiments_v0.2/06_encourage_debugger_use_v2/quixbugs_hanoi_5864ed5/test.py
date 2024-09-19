from hanoi import hanoi

def test__hanoi():
    """The mutant changes the destination peg from 'end' to 'helper' in the steps, which results in an invalid output."""
    output = hanoi(2, 1, 3)
    expected_output = [(1, 2), (1, 3), (2, 3)]
    assert output == expected_output, f"Expected {expected_output} but got {output}"