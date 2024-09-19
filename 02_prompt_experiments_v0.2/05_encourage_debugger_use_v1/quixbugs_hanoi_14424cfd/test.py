from hanoi import hanoi

def test__hanoi():
    """The mutant changes a critical step in the hanoi function, which alters the series of moves that are returned."""
    output = hanoi(2, 1, 3)
    expected_output = [(1, 2), (1, 3), (2, 3)]
    assert output == expected_output, f"Expected output {expected_output}, but got {output}"