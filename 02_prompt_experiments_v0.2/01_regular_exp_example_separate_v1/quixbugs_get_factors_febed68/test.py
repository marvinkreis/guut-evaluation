from get_factors import get_factors

def test__get_factors():
    """The mutant will return an empty list for prime factors like 101, while the correct implementation will return the prime itself."""
    output = get_factors(101)
    assert len(output) > 0, "get_factors must return factors for prime numbers"
    assert 101 in output, "get_factors must include the prime number in its factors"