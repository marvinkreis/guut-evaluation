from gcd import gcd  # Import the gcd function at the beginning

def test__gcd():
    """The mutant's change in parameter order causes the gcd function to fail."""
    output = gcd(35, 21)
    assert output == 7, "gcd must return the correct GCD for (35, 21)"