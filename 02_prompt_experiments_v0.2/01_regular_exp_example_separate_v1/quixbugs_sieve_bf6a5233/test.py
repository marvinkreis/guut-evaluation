from sieve import sieve

def test__sieve():
    """This test checks that the sieve function correctly identifies primes up to 10."""
    output = sieve(10)
    assert len(output) > 0, "sieve must detect prime numbers"
    assert output == [2, 3, 5, 7], "sieve must return the list [2, 3, 5, 7]"