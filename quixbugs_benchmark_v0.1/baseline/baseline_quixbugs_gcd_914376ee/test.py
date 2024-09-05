from gcd import gcd

def test_gcd():
    # The correct expected output for gcd(35, 21) is 7
    result = gcd(35, 21)
    assert result == 7, f"Expected 7, but got {result}"

    # This test case will help detect the mutant,
    # as the mutant gcd function will calculate gcd(21, 35)
    # which gives us an incorrect order of operations in the recursive calls.
    result_mutant = gcd(21, 35)
    # The correct gcd(21, 35) still results in 7, however,
    # it can be used to ensure the logic around order is functioning.
    assert result_mutant == 7, f"Expected 7, but got {result_mutant}"