from kheapsort import kheapsort

def test__kheapsort():
    """The mutant changes the loop to iterate over the entire array, causing incorrect output."""
    arr = [7, 6, 5, 4, 3, 2, 1]
    k = 3
    output = list(kheapsort(arr, k))
    assert output == [4, 3, 2, 1, 5, 6, 7], "kheapsort must yield the sorted output."