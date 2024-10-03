from next_permutation import next_permutation

def test__next_permutation_kill_mutant():
    """
    Test that verifies the next permutation generation. The input [1, 3, 2] has a valid next permutation, which 
    should be [2, 1, 3]. The mutant's altered condition results in returning None instead of a valid output,
    distinguishing its behavior from the baseline function.
    """
    perm = [1, 3, 2]
    output = next_permutation(perm)
    print(f"output for input {perm} = {output}")
    assert output == [2, 1, 3], f"Expected [2, 1, 3] but got {output}"