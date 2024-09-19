from next_permutation import next_permutation

def test__next_permutation():
    """Changing 'if perm[i] < perm[j]' to 'if perm[j] < perm[i]' would cause wrong permutations."""
    output = next_permutation([3, 2, 4, 1])
    expected_output = [3, 4, 1, 2]
    assert output == expected_output, f"expected {expected_output}, but got {output}"