from to_base import to_base

def test__to_base():
    """The mutant incorrectly changes the order of concatenation, which will produce reversed results."""
    output = to_base(31, 16)
    assert output == '1F', "to_base should correctly convert 31 to base 16 as '1F'"