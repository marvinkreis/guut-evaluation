from powerset import powerset

def test__powerset():
    """The mutant doesn't correctly generate the full power set, as it omits many combinations."""
    output = powerset(['a', 'b', 'c'])
    assert len(output) == 8, "powerset must return 8 subsets"
    assert output == [[], ['c'], ['b'], ['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['a', 'b', 'c']], "powerset must correctly generate all combinations"