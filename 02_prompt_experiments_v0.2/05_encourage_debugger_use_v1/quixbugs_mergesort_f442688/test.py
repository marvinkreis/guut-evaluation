from mergesort import mergesort

def test__mergesort():
    """Changing the base case from 'len(arr) <= 1' to 'len(arr) == 0' will cause incorrect behavior."""
    input_list = [3, 1, 4, 2]
    output = mergesort(input_list)
    assert output == [1, 2, 3, 4], "mergesort must sort the list correctly"