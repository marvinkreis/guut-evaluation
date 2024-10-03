from powerset import powerset

def test__powerset():
    """
    Test to ensure that the powerset function returns a complete power set with the empty set included.
    The input ['a', 'b', 'c'] should yield a power set that includes the empty set.
    The mutant is expected to fail this test as it does not return the empty set in the output.
    """
    output = powerset(['a', 'b', 'c'])
    assert output == [[], ['c'], ['b'], ['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['a', 'b', 'c']]