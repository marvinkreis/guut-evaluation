from subsequences import subsequences

def test__subsequences():
    """Changing the return value from [[]] to [] in subsequences causes it to not return any valid sequences."""
    output = subsequences(1, 5, 3)
    assert len(output) > 0, "subsequences must return valid ascending sequences"