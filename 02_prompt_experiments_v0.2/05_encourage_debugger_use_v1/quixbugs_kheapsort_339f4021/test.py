from kheapsort import kheapsort

def test__kheapsort():
    """The mutant version of kheapsort incorrectly yields elements, causing duplicates and incorrect sorting."""
    output = list(kheapsort([1, 2, 3, 5, 4], 1))
    assert output == [1, 2, 3, 4, 5], "kheapsort must return the sorted output without duplicates."