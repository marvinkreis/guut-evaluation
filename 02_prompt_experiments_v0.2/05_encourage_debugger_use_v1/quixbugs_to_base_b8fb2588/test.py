from to_base import to_base

def test__to_base():
    """Changing the result concatenation from 'result = alphabet[i] + result' to 'result = result + alphabet[i]' will reverse the output."""
    output = to_base(31, 16)
    assert output == '1F', "The conversion of 31 to base 16 should be '1F'."