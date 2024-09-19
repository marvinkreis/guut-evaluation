from pascal import pascal

def test__pascal():
    """The mutant incorrectly modifies loop boundaries causing an IndexError or incomplete rows."""
    output = pascal(5)
    assert output == [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]], "pascal must return valid rows for n=5"
    
    for n in range(2, 8):
        output = pascal(n)
        assert len(output) == n, f"pascal must return {n} rows for n={n}"
        assert output[-1][-1] == 1, f"The last element of the last row of pascal should be 1 for n={n}"