from next_permutation import next_permutation

def test_next_permutation():
    """Test to ensure that the mutant fails to produce the correct next permutation."""
    output = next_permutation([1, 2, 3])
    assert output == [1, 3, 2], "Next permutation should be [1, 3, 2]"