from subsequences import subsequences

def test__subsequences():
    """Changing the return value for k == 0 from [[]] to [] would lead to the wrong result."""
    output = subsequences(1, 5, 0)
    assert output == [[]], "Expected output should be [[]] when k is 0."