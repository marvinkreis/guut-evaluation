from gcd import gcd

def test__gcd():
    assert gcd(35, 21) == 7, "gcd(35, 21) should return 7"
    assert gcd(21, 35) == 7, "gcd(21, 35) should also return 7"