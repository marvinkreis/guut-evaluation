from sieve import sieve

def test__sieve():
    output = sieve(2)
    assert len(output) > 0, "sieve must detect prime numbers"
    assert output == [2], "sieve must correctly return the list of primes"