from subsequences import subsequences

def test__subsequences():
    """Changing the return value of subsequences for k == 0 from [[]] to [] causes the function to fail."""
    output = subsequences(1, 5, 0)
    assert output == [[]], "subsequences must return [[]] for k=0"