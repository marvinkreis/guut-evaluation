from gcd import gcd

def test__gcd_kill_mutant():
    """
    Test whether the computation of the GCD changes based on the order of arguments.
    The inputs (60, 48) are chosen because they should yield a clear result of 12 in the baseline,
    but the mutant implementation should hit an infinite recursion due to incorrect argument order.
    """
    a, b = 60, 48
    output = gcd(a, b)
    print(f"GCD of {a} and {b} is {output}")