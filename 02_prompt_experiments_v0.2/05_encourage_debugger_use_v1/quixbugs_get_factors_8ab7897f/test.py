from get_factors import get_factors

def test__get_factors():
    """The mutant improperly returns an empty list instead of the expected prime factors."""
    output = get_factors(4)
    assert len(output) > 1, "get_factors must return multiple instances of the prime factor 2 for input 4"