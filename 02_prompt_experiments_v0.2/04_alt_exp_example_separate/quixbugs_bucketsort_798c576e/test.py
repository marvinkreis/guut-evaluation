from bucketsort import bucketsort

def test__bucketsort():
    """The mutant changes the iteration from counts to arr, leading to incorrect outputs."""
    output = bucketsort([3, 1, 2, 3, 0], 4)
    assert output == [0, 1, 2, 3, 3], "bucketsort must return the correctly sorted array"