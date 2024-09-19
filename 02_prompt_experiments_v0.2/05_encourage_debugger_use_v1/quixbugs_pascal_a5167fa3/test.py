from pascal import pascal

def test__pascal():
    """The mutant fails due to an IndexError while generating Pascal's Triangle with n=5."""
    output = pascal(5)
    assert len(output) == 5, "Output should contain 5 rows."
    assert all(len(row) == i + 1 for i, row in enumerate(output)), "Each row should have increasing lengths."