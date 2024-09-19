from kheapsort import kheapsort

def test__kheapsort():
    """The mutant fails to sort correctly and yields duplicates for the input [4, 3] with k = 2."""
    output = list(kheapsort([4, 3], 2))
    expected_output = [3, 4]
    assert output == expected_output, f"Expected {expected_output} but got {output}"