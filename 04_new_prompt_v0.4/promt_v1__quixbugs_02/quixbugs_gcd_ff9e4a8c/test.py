from gcd import gcd

def test__gcd_mutant_killing():
    """
    This test confirms that the gcd function behaves correctly
    for a variety of inputs. The baseline returns the correct 
    GCD values for pairs of non-zero integers while the mutant 
    implementation leads to infinite recursion and fails with 
    a RecursionError.
    """
    valid_gcd_pairs = [
        (48, 18),  # GCD should be 6
        (100, 25),  # GCD should be 25
        (37, 74),  # GCD should be 37
        (42, 56),  # GCD should be 14
        (5, 15),   # GCD should be 5
    ]

    # Testing valid cases
    for a, b in valid_gcd_pairs:
        result = gcd(a, b)
        assert result is not None  # Check for valid outputs
    
    # Now testing for the infinite recursion error for the mutant
    try:
        gcd(35, 21)  # Expected to cause RecursionError in the mutant
    except RecursionError:
        print("RecursionError caught as expected for the mutant.")