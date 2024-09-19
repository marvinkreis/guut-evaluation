from subsequences import subsequences

def test__subsequences():
    """The mutant incorrectly removes the base case that allows the calculation of valid subsequences."""
    output = subsequences(a=1, b=5, k=3)
    assert len(output) > 0, "subsequences should return valid subsets of length k"