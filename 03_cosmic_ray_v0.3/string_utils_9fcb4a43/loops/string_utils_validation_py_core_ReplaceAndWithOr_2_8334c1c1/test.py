from string_utils.validation import is_decimal

def test__is_decimal():
    """The mutant incorrectly identifies non-decimal strings as decimal."""
    # Valid decimals
    assert is_decimal('42.0') == True, "Expected '42.0' to be identified as a decimal."
    assert is_decimal('3.14') == True, "Expected '3.14' to be identified as a decimal."
    
    # Invalid decimal checks
    assert is_decimal('42') == False, "Expected '42' to NOT be identified as a decimal."
    assert is_decimal('hello') == False, "Expected 'hello' to NOT be identified as a decimal."
    assert is_decimal('1.2.3') == False, "Expected '1.2.3' to NOT be identified as a decimal."
    assert is_decimal('not.a.number') == False, "Expected 'not.a.number' to NOT be identified as a decimal."