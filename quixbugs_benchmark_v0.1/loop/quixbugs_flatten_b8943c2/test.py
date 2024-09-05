from flatten import flatten

def test__flatten():
    output = list(flatten([[1, [], [2, 3]], [[4]], 5]))
    assert output == [1, 2, 3, 4, 5], "flatten must yield non-list elements correctly"