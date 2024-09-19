from powerset import powerset

def test__powerset():
    """Changing the return statement in powerset would cause it to output only the full set instead of the complete power set."""
    output = powerset(['a', 'b', 'c'])
    # The length should be 2^n for a set of size n (which is 3 here), so it should be 8
    assert len(output) == 8, "powerset must contain 8 subsets for input ['a', 'b', 'c']"