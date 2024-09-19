from bucketsort import bucketsort

def test__bucketsort():
    """Changing the loop to iterate over arr instead of counts causes incorrect sorting."""
    # Input with expected sorted order
    test_arr = [4, 2, 3, 1, 0, 4, 2]
    k = 5
    output = bucketsort(test_arr, k)
    assert output == [0, 1, 2, 2, 3, 4, 4], "bucketsort must sort the array correctly"