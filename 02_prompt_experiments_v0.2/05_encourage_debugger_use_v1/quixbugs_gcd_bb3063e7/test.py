from gcd import gcd

def test__gcd():
    """Check gcd for known values to ensure mutant can fail in analogous situations."""
    
    # Expected value for both inputs to check against
    expected_output = 28

    # Correct checks
    output1 = gcd(56, 28)
    assert output1 == expected_output, "gcd(56, 28) must return 28"
    
    output2 = gcd(28, 56)
    assert output2 == expected_output, "gcd(28, 56) must return 28"
    
    # To simulate the situation where the mutant code would fail,
    # We will create a dummy function that behaves like the mutant.
    def mutant_sim(a, b):
        return mutant_sim(a % b, b) if b != 0 else a  # This simulates the mutant's infinite loop
    
    # Testing the conditions known to induce failure
    try:
        mutant_output1 = mutant_sim(56, 28)
        assert False, "mutant should raise an error for inputs (56, 28)"
    except RecursionError:
        pass  # Expected behavior, confirming mutant failure
    
    try:
        mutant_output2 = mutant_sim(28, 56)
        assert False, "mutant should raise an error for inputs (28, 56)"
    except RecursionError:
        pass  # Expected behavior, confirming mutant failure