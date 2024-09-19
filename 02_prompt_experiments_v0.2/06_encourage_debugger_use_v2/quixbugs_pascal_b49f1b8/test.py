from pascal import pascal

def test__pascal():
    """Changing the range in the inner loop causes an IndexError and incomplete rows."""
    output = pascal(5)
    assert len(output) == 5, "Length of output must be 5 rows for input n=5"
    for row in output:
        assert row[-1] == 1, "Each row in Pascal's triangle should end with 1"