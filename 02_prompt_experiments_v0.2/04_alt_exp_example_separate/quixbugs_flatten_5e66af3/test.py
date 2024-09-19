from flatten import flatten

def test__flatten():
    """The mutant changes 'yield x' to 'yield flatten(x)', which causes the result to be invalid for any input nested list."""
    output = list(flatten([[1, [], [2, 3]], [[4]], 5]))
    assert output == [1, 2, 3, 4, 5], "flatten must yield all non-list objects"

    output2 = list(flatten([[1, 2], [3, [4, 5]], 6]))
    assert output2 == [1, 2, 3, 4, 5, 6], "flatten must yield all non-list objects in a new list"