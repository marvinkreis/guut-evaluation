from mergesort import mergesort

def test__mergesort_kill_mutant():
    """
    Test mergesort with a single-element list to ensure the function can handle this 
    properly. The input [7] should return [7] in both the Baseline and the Mutant, but 
    will fail in the Mutant due to improper base case handling.
    """
    output = mergesort([7])
    assert output == [7], f"Expected [7], but got {output}"