from lis import lis

def test__longest_increasing_subsequence():
    output = lis([4, 1, 5, 3, 7, 6, 2])
    assert output == 3, "The length of the longest increasing subsequence should be 3."