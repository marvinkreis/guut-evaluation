from sieve import sieve

def test__sieve():
    # Test case for the sieve function
    # We expect the primes up to 10 to be [2, 3, 5, 7]
    expected_primes = [2, 3, 5, 7]
    result = sieve(10)
    
    assert result == expected_primes, f"Expected {expected_primes}, got {result}"

    # Additional test case to ensure detection of the mutant:
    # We expect the primes up to 1 to be []
    expected_primes_empty = []
    result_empty = sieve(1)

    assert result_empty == expected_primes_empty, f"Expected {expected_primes_empty}, got {result_empty}"