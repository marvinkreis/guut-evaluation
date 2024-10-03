from pascal import pascal

def test__pascal():
    """
    Test the function pascal for n = 4. The expected output is
    [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]. The mutant will produce
    an incorrect row because it will not iterate to generate the correct
    last element, thus resulting in an incomplete last row.
    """
    n = 4
    output = pascal(n)
    print(f"output = {output}")
    assert output == [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]  # Expect this for Baseline