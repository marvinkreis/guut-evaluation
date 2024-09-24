from string_utils.validation import is_decimal

def test__is_decimal():
    assert is_decimal('42.0') == True  # This should be a valid decimal
    assert is_decimal('42') == False    # This should not be recognized as a decimal
    assert is_decimal('42.00') == True  # This should also be a valid decimal
    assert is_decimal('not_a_number') == False  # This should not be recognized as a decimal
    assert is_decimal('-3.14') == True  # This should be valid decimal too
    assert is_decimal('3.14a') == False  # This should not be a valid decimal