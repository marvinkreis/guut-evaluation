from lis import lis

def test__lis():
    """
    Verify that the longest increasing subsequence function computes the correct length
    of the longest increasing subsequence for a known input. The test input is [4, 1, 5, 3, 7, 6, 2],
    which has a longest increasing subsequence of length 3. 
    The mutant should fail this test because its modification leads to an incorrect calculation.
    """
    arr = [4, 1, 5, 3, 7, 6, 2]
    output = lis(arr)
    assert output == 3, f"Expected output of 3, but got {output}"