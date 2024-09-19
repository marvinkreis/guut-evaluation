from to_base import to_base

def test__to_base():
    """The mutant's change in concatenating order causes incorrect results."""
    output = to_base(255, 36)  # Expecting '73'
    assert output == '73', f"Expected '73' but got '{output}'"