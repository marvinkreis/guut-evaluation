from sieve import sieve

def test__sieve():
    """The mutant changes 'all' to 'any' in sieve, causing it to always return an empty list."""
    output = sieve(5)
    assert len(output) > 0, "sieve must detect prime numbers"