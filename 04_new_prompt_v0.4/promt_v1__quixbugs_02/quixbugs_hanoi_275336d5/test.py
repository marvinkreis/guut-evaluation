from hanoi import hanoi

def test__hanoi():
    """
    Test the Towers of Hanoi function with a height of 2.
    The baseline should generate the output [(1, 2), (1, 3), (2, 3)],
    while the mutant will generate a different sequence.
    This test serves to ensure that the expected sequence of moves is produced
    according to the defined problem rules.
    """
    expected_output = [(1, 2), (1, 3), (2, 3)]
    output = hanoi(2, 1, 3)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"