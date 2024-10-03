from sieve import sieve

def test__sieve_mutant_killing():
    """
    This test case aims to demonstrate the behavior of the sieve function
    with prime number inputs. The baseline should return the correct list of
    prime numbers, while the mutant should produce an empty list, thus killing
    the mutant.
    """
    prime_check = 11  # Known prime number

    output = sieve(prime_check)
    print(f"max = {prime_check}, primes = {output}")
    assert output == [2, 3, 5, 7, 11], "The mutant should return an empty list."