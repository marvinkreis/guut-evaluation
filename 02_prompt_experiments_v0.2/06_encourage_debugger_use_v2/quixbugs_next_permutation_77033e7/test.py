from next_permutation import next_permutation

def test__next_permutation():
    """Changing the order of comparison in next_permutation leads to incorrect results."""
    output = next_permutation([3, 2, 4, 1])
    assert output == [3, 4, 1, 2], "next_permutation must produce the correct next permutation"