from get_factors import get_factors

def test__get_factors_mutant_killing():
    """
    Test whether the function correctly identifies the prime factorization of prime numbers.
    Specifically, for n = 11, the baseline should return [11], while the mutant should return an empty list [].
    This test confirms that the mutant does not handle prime factors properly.
    """
    assert get_factors(11) == [11], "Expected [11] for a prime number input."