from kheapsort import kheapsort

def test__kheapsort():
    """The mutant version of kheapsort does not correctly sort the input array."""
    input_data = [3, 2, 1, 5, 4]
    k = 2
    output = list(kheapsort(input_data, k))
    assert output == sorted(input_data), "kheapsort must sort the array correctly"