from powerset import powerset

def test__powerset():
    output = powerset(['a', 'b', 'c'])
    assert output == [[], ['c'], ['b'], ['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['a', 'b', 'c']], "powerset should return all subsets"