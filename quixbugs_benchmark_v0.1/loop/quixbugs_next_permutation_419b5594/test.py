from next_permutation import next_permutation

def test__next_permutation():
    output = next_permutation([3, 2, 4, 1])
    assert output == [3, 4, 1, 2], "next_permutation must return the correct next permutation."