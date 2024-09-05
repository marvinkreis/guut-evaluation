from mergesort import mergesort

def test__mergesort():
    output = mergesort([1])
    assert output == [1], "mergesort must return the sorted list containing a single element"