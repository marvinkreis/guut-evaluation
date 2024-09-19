from next_permutation import next_permutation

def test_next_permutation():
    """Changing the comparison operator in next_permutation leads to incorrect results."""
    output = next_permutation([3, 2, 4, 1])
    assert output == [3, 4, 1, 2], f"Expected next_permutation([3, 2, 4, 1]) to be [3, 4, 1, 2], but got {output}."