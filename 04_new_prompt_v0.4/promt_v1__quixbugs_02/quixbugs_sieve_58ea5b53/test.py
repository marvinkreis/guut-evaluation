from sieve import sieve

def test__sieve_kills_mutant():
    """
    Test the sieve function with an input max=15. The expected output is 
    the list of primes up to 15: [2, 3, 5, 7, 11, 13]. This test is designed 
    to fail on the mutant implementation, which should return an empty list,
    demonstrating its failure to compute primes correctly.
    """
    output = sieve(15)  # Testing with max=15
    assert output == [2, 3, 5, 7, 11, 13], f"Expected primes up to 15 but got {output}"