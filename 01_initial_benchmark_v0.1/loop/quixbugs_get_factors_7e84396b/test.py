from get_factors import get_factors

def test__get_factors():
    output = get_factors(100)
    assert output == [2, 2, 5, 5], "get_factors must return the correct prime factors with repetitions"