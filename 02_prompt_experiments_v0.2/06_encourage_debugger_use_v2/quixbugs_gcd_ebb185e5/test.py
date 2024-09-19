from gcd import gcd

def test__gcd():
    """The mutant changes the recursive call order, which leads to maximum recursion depth exceeded for certain inputs."""
    output = gcd(35, 21)
    assert output == 7, "gcd(35, 21) must return 7"