from kth import kth

def test__kth():
    """The mutant alters the behavior of kth 
    by not correctly computing the index when k >= num_lessoreq.
    Calling kth([3, 1, 2, 4, 5], 4) should return 5."""
    arr = [3, 1, 2, 4, 5]
    k = 4
    output = kth(arr, k)
    assert output == 5, "kth must return the 5th lowest element of the array"