from bucketsort import bucketsort

def test__bucketsort():
    """The mutant changes the output construction causing incorrect sorting and length of the result."""
    arr = [3, 1, 4, 1, 5, 9]
    k = 10
    output = bucketsort(arr, k)
    
    # The output should be sorted and must match the expected length
    assert len(output) == 6, "Output length must match input length"
    assert output == [1, 1, 3, 4, 5, 9], "Output must be a correctly sorted list"