from sieve import sieve

def test__sieve():
    """Changing 'all' to 'any' in sieve causes it to always return an empty list instead of the correct primes."""
    # Test that sieve(5) produces a non-empty result
    output = sieve(5)
    assert len(output) > 0, "sieve must detect prime numbers up to 5"
    
    # Test that sieve(3) also produces a non-empty result
    output = sieve(3)
    assert len(output) > 0, "sieve must detect prime numbers up to 3"