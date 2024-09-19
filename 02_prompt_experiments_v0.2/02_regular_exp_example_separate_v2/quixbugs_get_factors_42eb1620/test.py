from get_factors import get_factors

def test__get_factors():
    """The mutant returns [] for prime inputs, while the correct implementation returns [n]."""
    output = get_factors(101)
    assert len(output) > 0, "get_factors must return prime factors for prime inputs"