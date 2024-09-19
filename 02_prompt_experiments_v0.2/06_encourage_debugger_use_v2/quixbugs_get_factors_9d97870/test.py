from get_factors import get_factors

def test__get_factors():
    """Changing the return value for prime cases from [n] to [] would cause the function to fail for prime numbers."""
    output = get_factors(101)
    assert len(output) > 0, "get_factors must return prime factors for prime numbers"