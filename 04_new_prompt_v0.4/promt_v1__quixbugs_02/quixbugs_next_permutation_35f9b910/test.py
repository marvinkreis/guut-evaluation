from next_permutation import next_permutation

def test__next_permutation_killing():
    """
    Test the next_permutation function with an input that should produce a valid next permutation.
    The input [1, 2, 3] should result in [1, 3, 2]. The mutant will fail to produce this correct
    output due to logic inversion in the comparison.
    This test verifies that the mutant is indeed faulty by asserting the expected behavior.
    """
    output = next_permutation([1, 2, 3])
    assert output == [1, 3, 2], f"Expected [1, 3, 2], but got {output}"