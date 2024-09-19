from powerset import powerset

def test__powerset():
    """Changing the return statement caused the empty subset to be omitted from the result."""
    output = powerset(['a', 'b'])
    assert [] in output, "The empty subset should be included in the powerset output"