from lis import lis

def test__lis():
    """Mutant changes the calculation of the longest increasing subsequence leading to incorrect output."""
    output = lis([3, 10, 2, 1, 20])
    assert output == 3, "The output must be 3 for the longest increasing subsequence"