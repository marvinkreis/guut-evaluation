from gcd import gcd

def test__gcd():
    """This test ensures that the mutant's incorrect argument order is detected."""
    # Test with gcd(0, 5)
    try:
        output1 = gcd(0, 5)
        assert output1 == 5, "gcd(0, 5) should return 5"
    except RecursionError:
        assert False, "Mutant detected: RecursionError for gcd(0, 5)"
    
    # Test with gcd(5, 0)
    try:
        output2 = gcd(5, 0)
        assert output2 == 5, "gcd(5, 0) should return 5"
    except RecursionError:
        assert False, "Mutant detected: RecursionError for gcd(5, 0)"

# Running the test to verify behavior
test__gcd()