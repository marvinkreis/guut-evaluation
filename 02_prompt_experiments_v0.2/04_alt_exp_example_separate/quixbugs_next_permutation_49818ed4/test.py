from next_permutation import next_permutation

def test__next_permutation():
    """The mutant fails to find the next permutation for [1, 5, 4, 3, 2]."""
    output = next_permutation([1, 5, 4, 3, 2])
    assert output is not None, "next_permutation must produce a valid next permutation"