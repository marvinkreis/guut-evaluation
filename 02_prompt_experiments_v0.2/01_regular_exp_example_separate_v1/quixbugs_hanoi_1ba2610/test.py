from hanoi import hanoi

def test__hanoi():
    """The mutant version of hanoi incorrectly changes a move from (start, end) to (start, helper)."""
    correct_output = [(1, 2), (1, 3), (2, 3)]
    output = hanoi(2)
    assert output == correct_output, f"Expected steps {correct_output}, got {output}"