from subsequences import subsequences

def test__subsequences():
    """Changing the return value from [[]] to [] in subsequences will cause it to return no valid sequences."""
    output = subsequences(1, 5, 3)
    assert len(output) > 0, "subsequences must generate valid ascending subsequences"