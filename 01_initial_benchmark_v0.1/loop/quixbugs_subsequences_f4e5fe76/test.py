from subsequences import subsequences

def test__subsequences_k_equals_zero():
    output = subsequences(1, 5, 0)
    assert output == [[]], "subsequences must return [[]] when k=0"