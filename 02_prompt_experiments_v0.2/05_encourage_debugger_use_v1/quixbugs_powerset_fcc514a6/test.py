from powerset import powerset

def test__powerset():
    """The mutant fails because it does not return all subsets, including the empty set."""
    output = powerset(['a', 'b', 'c'])
    expected_output = [[], ['c'], ['b'], ['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['a', 'b', 'c']]
    assert output == expected_output, f"Expected {expected_output}, but got {output}"