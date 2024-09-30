from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether encoding the integer 400 to a Roman numeral returns 'CD'. The mutant alters the Roman numeral mapping
    for hundreds, causing it to result in a KeyError when accessing key '5'. This test confirms the correct behavior 
    of the baseline code while invoking an error in the mutant.
    """
    output = roman_encode(400)
    assert output == 'CD', f"Expected 'CD' but got {output}"