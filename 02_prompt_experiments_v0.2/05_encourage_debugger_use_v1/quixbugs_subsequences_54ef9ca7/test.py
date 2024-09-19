from subsequences import subsequences

def test__subsequences():
    """Changing return [[]] to return [] in subsequences will cause it to yield no valid sequences for k > 0."""
    output = subsequences(1, 5, 3)
    assert len(output) > 0, "subsequences must return valid sequences for k > 0"