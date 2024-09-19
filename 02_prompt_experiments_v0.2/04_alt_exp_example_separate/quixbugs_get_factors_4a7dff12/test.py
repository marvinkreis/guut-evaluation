from get_factors import get_factors

def test__get_factors():
    """Changing the return value in get_factors from [n] to an empty list would cause it to fail for prime inputs."""
    output = get_factors(101)
    assert len(output) > 0, "get_factors must return factors for prime numbers"