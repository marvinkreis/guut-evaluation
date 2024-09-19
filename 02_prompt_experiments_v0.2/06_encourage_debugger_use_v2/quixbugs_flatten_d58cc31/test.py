from flatten import flatten

def test__flatten():
    """The mutant incorrectly yields generator objects instead of the actual flattened output."""
    output = list(flatten([[1, [], [2, 3]], [[4]], 5]))
    assert output == [1, 2, 3, 4, 5], "flatten must return the expected flattened list"
    
    # Additional check with different input
    output = list(flatten([1, [2, 3], [4, [5]], 6]))
    assert output == [1, 2, 3, 4, 5, 6], "flatten must correctly flatten more complex nested lists"