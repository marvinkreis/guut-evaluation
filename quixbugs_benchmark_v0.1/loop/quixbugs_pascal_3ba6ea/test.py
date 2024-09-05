from pascal import pascal

def test__pascal():
    output = pascal(5)
    assert output == [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]], "pascal must return the correct first five rows of Pascal's triangle"