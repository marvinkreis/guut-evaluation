from gcd import gcd

def test__gcd():
    """The mutant has incorrect recursion logic that causes maximum recursion depth exceeded."""
    assert gcd(35, 21) == 7, "gcd must return the correct GCD"
    assert gcd(48, 18) == 6, "gcd must return the correct GCD"
    assert gcd(100, 25) == 25, "gcd must return the correct GCD"
    assert gcd(10, 5) == 5, "gcd must return the correct GCD"