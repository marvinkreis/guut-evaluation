from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    """The mutant changes the condition in is_ip_v4, causing it to fail with None as input."""
    output = is_ip_v4(None)
    assert output == False, "is_ip_v4 must return False for None input"