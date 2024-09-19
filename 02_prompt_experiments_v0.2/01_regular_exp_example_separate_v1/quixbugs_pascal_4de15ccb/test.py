from pascal import pascal

def test__pascal():
    """The mutant fails to generate the complete last elements of each row of Pascal's triangle."""
    expected_output = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    output = pascal(5)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"