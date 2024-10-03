from powerset import powerset

def test__powerset_mutant_killer():
    """
    Test the powerset function with input ['a', 'b', 'c']. The expected output includes all subsets of the input list.
    The Baseline produces multiple subsets, while the Mutant only outputs the full set, demonstrating distinct behavior. 
    """
    output = powerset(['a', 'b', 'c'])
    expected_output = [[], ['c'], ['b'], ['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['a', 'b', 'c']]
    assert output == expected_output