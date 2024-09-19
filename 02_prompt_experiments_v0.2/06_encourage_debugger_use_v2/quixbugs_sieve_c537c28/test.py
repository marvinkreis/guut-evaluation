from sieve import sieve

def test__sieve():
    """The mutant changes 'all' to 'any' in sieve, resulting in no prime numbers being found."""
    output = sieve(5)
    assert len(output) > 0, "sieve must detect prime numbers"