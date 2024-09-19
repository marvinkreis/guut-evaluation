from flatten import flatten

def test__flatten():
    """The mutant changes yield behavior, causing it to incorrectly return generator objects instead of individual values."""
    
    # Test with a nested list containing numbers
    output = list(flatten([[1, [], [2, 3]], [[4]], 5]))
    assert output == [1, 2, 3, 4, 5], "Output should be all non-list objects flattened into a single list"
    
    # Test with a single non-list item
    output_single = list(flatten([5]))
    assert output_single == [5], "Output should yield the non-list integer directly"